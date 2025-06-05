#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include <algorithm>

using namespace std;
#define __DEBUG_DumpFiles 1
#define __DEBUG_Progress 1

class BBox {
  public:
  int x1, x2, y1, y2;
  BBox() {
    x1=y1=10000;
    x2=y2=0;
  }  
  void update(int x,int y) {
    if(x<x1) x1=x;
    if(x>x2) x2=x;
    if(y<y1) y1=y;
    if(y>y2) y2=y;
  }
  void print(FILE *fp=stdout) {
    fprintf(fp,"[%d,%d]--[%d,%d]: %d,%d",x1,y1,x2,y2, W(), H());
  }
  int W() {
    return (x2-x1+1);
  }
  int H() {
    return (y2-y1+1);
  }
};

class Image {
    public:
    int W=0, H=0;
    bool isYuvFormat=false;
    vector<float> image;

    Image() {
        W = 0;
        H = 0;
    }

    Image(int iW, int iH, int *arr) {
        copyFromIntArray(iW, iH, arr);
    }

    void copyFromIntArray(int iW, int iH, int *arr) {
        clear();
        W = iW;
        H = iH;
        // YUV format is passed only Y part is copied
        isYuvFormat = false;
        image = vector<float>(W*H);
        for(int n=0; n<W*H; n++) image[n] = arr[n];
    }

    void copy(Image &ref, int yuvFormat=false) {
        clear();
        W = ref.W;
        H = ref.H;
        int yuv = 0;
        if(yuvFormat) {
            isYuvFormat=true;
            yuv = W*H/2;
        }
        image = vector<float>(W*H+yuv);
        #if __DEBUG_Progress>0
        cout<<"Copy image "<<W<<", "<<H<<endl;
        #endif
        for(int n=0;n<W*H;n++) image[n] = ref.image[n];
        for(int n=0;n<yuv;n++) image[W*H+n] = 128;
    }

    Image(int iW, int iH, float dval=0) {
        init(iW, iH, dval);
    }

    ~Image() {
        #if __DEBUG_Progress>0
        cout<<"clear Image"<<endl;
        #endif
        clear();
    }

    void init(int iW, int iH, float dval=0) {
        W = iW;
        H = iH;
        image = vector<float>(W*H);
        #if __DEGBUG_Progress>0
        cout<<"Create image "<<W<<", "<<H<<":"<<dval<<endl;
        #endif
        setConst(dval);
    }

    void clip(int &nx, int &ny) {
        if (nx<0) nx = 0;
        if (nx>=W) nx = W-1;
        if (ny<0) ny = 0;
        if (ny>=H) ny = H-1;
    }

    float index(int nx, int ny) {
        clip(nx, ny);
        return image[nx + ny*W];
    }

    void set(int nx, int ny, float val) {
        clip(nx, ny);
        image[nx+ny*W] = val;
    }

    void resize(Image &dstImage, int nW, int nH) {
        dstImage.init(nW,nH,0);
        float wSc = (W-1)*1.0f/(nW-1);
        float hSc = (H-1)*1.0f/(nH-1);
        for(int r=0; r<nH; r++) {
            for(int c=0; c<nW; c++) {
                int sr = (int)(r*hSc);
                int sc = (int)(c*wSc);
                dstImage.set(c,r,index(sc,sr));
            }
        }
    }

    int nPixels() {
        int N = W * H;
        if(isYuvFormat) N += W*H*0.5;
        return N;
    }

    void adaptiveGaussianThreshold(int blockSize=11, int constant=5) {
        // Block size must be odd
        if (blockSize % 2 == 0) blockSize++;
        
        int halfSize = blockSize / 2;
        float sigma = blockSize / 3.0f;
        
        // Precompute Gaussian kernel
        float kernel[blockSize][blockSize];
        float sum = 0.0f;
        
        for (int dy = -halfSize; dy <= halfSize; dy++) {
            for (int dx = -halfSize; dx <= halfSize; dx++) {
                float val = exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
                kernel[dy + halfSize][dx + halfSize] = val;
                sum += val;
            }
        }
        
        // Normalize kernel
        for (int dy = 0; dy < blockSize; dy++) {
            for (int dx = 0; dx < blockSize; dx++) {
                kernel[dy][dx] /= sum;
            }
        }
        
        float tImage[W*H];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                // Calculate weighted sum
                float weightedSum = 0.0f;
                
                for (int dy = -halfSize; dy <= halfSize; dy++) {
                    for (int dx = -halfSize; dx <= halfSize; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            weightedSum += index(nx,ny) * kernel[dy + halfSize][dx + halfSize];
                        }
                    }
                }
                
                int threshold = (int)weightedSum - constant;
                tImage[x+y*W] = (index(x,y) > threshold) ? 255 : 0;
            }
        }

        for(int n=0; n<W*H; n++) image[n]=tImage[n];    
    }

    void clear() {
        image.clear();
        W = 0;
        H = 0;
    }

    void setConst(float data) {
        for(int n=0;n<W*H;n++) image[n]=data;
    }

    void fromFile(const char *fname) {
        FILE *fp = fopen(fname,"r");
        #if __DEBUG_Progress>0
        cout<<"read from file:"<<fname;
        #endif
        clear();
        fscanf(fp, "%d,%d", &W, &H);
        #if __DEBUG_Progress>0
        cout<<", W="<<W<<", H="<<H;
        #endif
        image = vector<float>(W*H);
        for(int n=0; n<W*H; n++) fscanf(fp,",%f",&image[n]);
        fclose(fp);
        #if __DEBUG_Progress>0
        cout<<" .... DONE"<<endl;
        #endif
    }

    void dump2File(const char *fname) {
        FILE *fp = fopen(fname,"w");
        fprintf(fp,"%d,%d",W,H);
        for(int n=0;n<W*H;n++) fprintf(fp,",%f",image[n]);
        fclose(fp);
        #if __DEBUG_Progress>0
        cout<<"dump2File: "<<fname<<endl;        
        #endif
    }

    void printSample(int offset, int len, FILE *fp=stdout) {
        cout<<"printSample: "<<offset<<", "<<len<<endl;
        for(int n=offset;n<offset+len;n++) {
            if(n>=W*H) break;
            fprintf(fp,"%f, ",image[n]);
        }
        fprintf(fp,"\n");
    }
};

class UnionFind {
    public:
    map<int,int> parent;
    map<int,int> rank;

    UnionFind() {
        parent.clear();
        rank.clear();
    }

    int find(int p, bool trace=false) {
        if (parent.find(p)==parent.end()) {
            parent[p] = p;
            rank[p] = 1;
        } else {
            if(parent[p]!=p) {
                parent[p] = find(parent[p]);
            }
        }
        if(trace) printf("UnionFind.find(%d)=%d",p,parent[p]);
        return parent[p];
    }
    
    bool connected(int p, int q, bool trace=false) {
        return find(p,trace) == find(q,trace);
    }

    void join(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);

        if (rootP != rootQ) {
            // Union by rank
            if (rank[rootP] > rank[rootQ]) {
                parent[rootQ] = rootP;
            } else {
                if (rank[rootP] < rank[rootQ]) {
                    parent[rootP] = rootQ;
                } else {
                    parent[rootQ] = rootP;
                    rank[rootP] += 1;
                }
            }
        }

    }
};

void testUnionFind() {
    printf("testUnionFind: START\n");
    UnionFind uf;
    uf.join(1, 2);
    uf.join(2, 5);
    uf.join(5, 6);
    uf.join(6, 7);
    uf.join(3, 8);
    uf.join(5, 9);

    printf("True %d\n",uf.connected(1, 2));
    printf("True %d\n",uf.connected(5, 7)); 
    printf("False %d\n",uf.connected(3, 9)); 
    printf("False %d\n",uf.connected(3, 1));
    printf("True %d\n",uf.connected(1, 9));
    printf("True %d\n",uf.connected(2, 7, true));
    printf("testUnionFind: DONE.\n");
}

int clip(int val, int max, int min=0) {
    if(val<min) return min;
    if(val>max) return max;
    return val;
}

class SpeedSignExtract {
    public:
    Image lImage;
    map<int,BBox> labelBBox;
    map<int,Image> images;
    int nW, nH;

    SpeedSignExtract() {
    }

    void connectedComponents(Image &bImage, float thresh=127) {
        UnionFind uf;
        int label = 0;
        lImage.init(bImage.W, bImage.H, -1);
        
        #if __DEBUG_Progress>0
        cout<<"scan thru image .."<<endl;
        #endif
        int nSkip = 0;
        for(int r=0; r<bImage.H; r++) {
            for(int c=0; c<bImage.W; c++) {
                if (bImage.index(c,r)>thresh) {
                    nSkip += 1;
                    continue;
                }
                
                vector<float> neigh;
                vector<int> neighLabel;
                // previous row
                int nr = clip(r-1,bImage.H-1);
                int c1 = clip(c-1,bImage.W-1);
                int c2 = clip(c+1,bImage.W-1);
                if(nr<r) {
                    for(int nc=c1; nc<=c2; nc++) {
                        neigh.push_back(bImage.index(nc, nr));
                        neighLabel.push_back(lImage.index(nc, nr));
                    }
                }

                // current row
                for(int nc=c1; nc<c; nc++) {
                    neigh.push_back(bImage.index(nc, r));
                    neighLabel.push_back(lImage.index(nc, r));
                }
                
                float nmin = *min_element(neigh.begin(), neigh.end());
                // printf("(%d,%d):%d, %1.2f | ",c,r,neigh.size(),nmin);
                if (nmin>=thresh) { 
                    // all neighbours in background
                    label += 1;
                    lImage.set(c,r,label);
                } else {
                    // find min label among neighbors
                    int minLabel = 1000;
                    for(int nc=0; nc<neigh.size(); nc++) {
                        if(neigh[nc]<thresh) {
                            if(neighLabel[nc]<minLabel) {
                                minLabel = neighLabel[nc];
                            }
                        }
                    }
                    // assign current pixel with min label (as of now)
                    lImage.set(c,r,minLabel);
                    // add label to unionFind
                    for(int nc=0; nc<neigh.size(); nc++) {
                        if(neighLabel[nc]>=0) uf.join(minLabel, neighLabel[nc]);
                    }
                }

                neigh.clear();
                neighLabel.clear();
            } // for:widht:c
        } // for:height:r
        #if __DEBUG_Progress>0
        cout<<"label="<<label<<", nSkip="<<nSkip<<endl;
        #endif

        #if __DEBUG_Progress>0
        cout<<"reduce to final labels .."<<endl;
        #endif
        // reassign image with final label
        for(int r=0; r<bImage.H; r++) {
            for(int c=0; c<bImage.W; c++) {

                if (lImage.index(c,r)<0) continue;
                int nLabel = uf.find(lImage.index(c,r));
                if(labelBBox.find(nLabel)==labelBBox.end()) labelBBox[nLabel]=BBox();
                labelBBox[nLabel].update(c,r);
                lImage.set(c,r,nLabel);

            } // for:width:c
        } // for:height:r

        #if __DEBUG_Progress>0
        for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) { 
            cout<<it->first<<":"; 
            it->second.print(); 
            cout<<endl; 
        }
        #endif
        
        #if __DEBUG_DumpFiles>0
        lImage.dump2File("tests/labelImageRaw.out");
        #endif
    }

    void extractImages(int padxy=2, int nW=28, int nH=28) {
        this->nW = nW;
        this->nH = nH;
        
        // initialize images for each label
        for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) {
            int label = it->first;
            if(label==-1) continue;
            int W = it->second.W() + padxy*2;
            int H = it->second.H() + padxy*2;
            images[label].init(W,H,0);
            #if __DEBUG_Progress>0
            cout<<"label:"<<label<<", "<<W<<", "<<H<<endl;
            #endif
        }
        // fill image for each label
        for(int r=0; r<lImage.H; r++) {
            for(int c=0; c<lImage.W; c++) {
                int label = lImage.index(c,r);
                if(label==-1) continue;
                int lc = c - labelBBox[label].x1 + padxy;
                int lr = r - labelBBox[label].y1 + padxy;
                images[label].set(lc,lr,255);
            }
        }
        // resize each label image
        for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) {
            int label = it->first;
            if(label==-1) continue;
            #if __DEBUG_Progress>0
            cout<<"resize label:"<<label<<endl;
            #endif
            Image fImage;
            images[label].resize(fImage, nW, nH);
            images[label].copy(fImage, true);
            #if __DEBUG_DumpFiles>0
            char fname[100];
            sprintf(fname,"tests/label_%d.out",label);
            images[label].dump2File(fname);
            #endif
            
        }
    }

    void flatten(vector<int> &flatData, int &nImages, int &w, int &h) {
        int N=0;
        nImages = images.size();
        w = nW;
        h = nH;
        for(map<int,Image>::iterator it=images.begin(); it!=images.end(); ++it) {
            N += it->second.nPixels();
        }
        
        flatData = vector<int>(N);
        int gn = 0;
        for(map<int,Image>::iterator it=images.begin(); it!=images.end(); ++it) {
            int lN = it->second.nPixels();
            cout<<it->first<<":"<<it->second.isYuvFormat<<";  ";
            for(int n=0; n<lN; n++) {
                flatData[gn] = (int)it->second.image[n];
                gn += 1;
            }
        }
        FILE *fp = fopen("tests/test_flatten.out","w");
        int noff = 0;
        for(int nimg=0;nimg<images.size();nimg++) {
            for(int n=0;n<nW*nH*1.5;n++) fprintf(fp,"%d,",flatData[noff++]);
        }
        fclose(fp);
    }
};

#define nv21dtype int
void exampleJniFunction(nv21dtype *imageArray, int W, int H, vector<int> &flattenImages, int &nImages, int &oW, int &oH) {
    Image image(W,H,imageArray); // create Image object
    SpeedSignExtract sobj; // initialize extraction object

    image.adaptiveGaussianThreshold(); // binarize the image
    sobj.connectedComponents(image, 127); // apply connected components
    sobj.extractImages(); // extract invidual images
    
    sobj.flatten(flattenImages, nImages, oW, oH); // get flattened YUV images
}

int main(int argc, char *argv[]) {

    printf("Usage: %s <mode> <input .dat> \n",argv[0]);
    printf("    mode: 0 for general processing, 1 for JNI simulation\n");
    printf("\n");

    if(argc<2) {
        return 0;
    }
    char mode = argv[1][0];
    cout<<"mode="<<mode<<endl;
    if (mode=='1') {
        FILE *fp = fopen(argv[2],"r");
        int w,h,*data;
        fscanf(fp,"%d,%d",&w,&h);
        data = new int[(int)(w*h*1.5)];
        for(int n=0;n<w*h*1.5;n++) fscanf(fp,",%d",&data[n]);
        fclose(fp);

        vector<int> flatImages;
        int nFlatImages, nW, nH;
        exampleJniFunction(data, w, h, flatImages, nFlatImages, nW, nH);
        printf("Flat image info: nImages=%d, W=%d, H=%d, %d\n",nFlatImages,nW,nH,flatImages.size());

        fp = fopen("tests/flatImages.out","w");
        int noff = 0;
        for(int nimg=0;nimg<nFlatImages;nimg++) {
            for(int n=0;n<nW*nH*1.5;n++) fprintf(fp,"%d,",flatImages[noff++]);
        }
        fclose(fp);

        return 0;
    }

    int w,h;
    Image image;
    // image.fromFile("../../data/SpeedSign/testImage123.dat");
    image.fromFile(argv[2]);
    // image.dump2File("tests/input_image.out");
    image.printSample(0,10);
    SpeedSignExtract sobj;
    
    map<int,int> root;
    root[1] = 10;
    root[100] = 20;
    cout<<root.size()<<", "<<root[1]<<endl;
    cout<<(root.find(11)==root.end())<<endl;
    root.clear();

    testUnionFind();

    image.adaptiveGaussianThreshold();
    image.dump2File("tests/adaptfilt_image.out");
    sobj.connectedComponents(image, 127);
    sobj.extractImages();
    
    cout<<"done.";
    return 0;
}