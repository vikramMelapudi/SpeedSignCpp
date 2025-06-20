#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include <algorithm>

using namespace std;
// #define __DEBUG_DumpFiles 1
// #define __DEBUG_Progress 1
#define __UseResizeLinear 1

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
  int size() {
    return W()*H();
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
        #if __UseResizeLinear>0
        resizeLinear(dstImage, nW, nH);
        return;
        #endif

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

    void resizeLinear(Image &dstImage, int nW, int nH) {
        // neighbour map
        const int cdel[] = {0,0,1,1};
        const int rdel[] = {0,1,0,1};

        dstImage.init(nW,nH,0);
        float wSc = (W-1)*1.0f/(nW-1);
        float hSc = (H-1)*1.0f/(nH-1);
        for(int r=0; r<nH; r++) {
            for(int c=0; c<nW; c++) {
                // location in source image
                float ir = r*hSc;
                float ic = c*wSc;
                // left, top most pixel in source image
                int sr = (int)ir;
                int sc = (int)ic;
                // compute distance weighted average pixel value
                float val = 0;
                float norm = 0;
                for(int nn=0; nn<4; nn++) {
                    int rr = sr+rdel[nn]; if(rr>=H) rr=H-1;
                    int cc = sc+cdel[nn]; if(cc>=W) cc=W-1;
                    float d = 2.0 - (abs(rr-ir)+abs(cc-ic)); // exact pixel map will have weight 2.0 and others <2.0
                    norm += d;
                    val += d*index(cc,rr);
                }
                // normalize with weights
                val = val/norm;

                // assign normalized weighted sum
                dstImage.set(c, r, val);
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

bool pairCompByValue(const pair<int,int>& a, const std::pair<int, int>& b) {
    return a.second < b.second;
}
bool pairCompByValueFloat(const pair<int,float>& a, const std::pair<int, float>& b) {
    return a.second < b.second;
}

class SpeedSignExtract {
    public:
    Image lImage; // region labelled image
    map<int,BBox> labelBBox; // bounding box of digits/object in the image
    map<int,Image> images; // extracted image
    int nW, nH; // final resize WxH for each image
    // filter settings (see constructor for details)
    float minHFrac, minWFrac, minARFrac, marginFrac, bigSizeFrac;
    int topN;

    SpeedSignExtract() {
        minHFrac = 0.28; // min height of digit as fraction of image
        minWFrac = 0.1; // min width of digit as fraction of image
        minARFrac = 1.1; // min aspect-ration (H/W) of digit as fraction of image
        marginFrac = 0.05; // margin boundary defined as fraction of image
        topN = 3; // max digit-boxes to return
        bigSizeFrac = 0.8; // reject too big digits as fraction of image
    }

    void connectedComponents(Image &bImage, float thresh=127) {
        // thresholed image (bImage) and threhold for binarization (thresh), 
        // NOTE: if pixel>thresh then omitted i.e. only dark regions are considered as digits/objects
        // outputs:
        //  1. this->lImage - each pixel has unique label
        //  2. this->labelBBox - bounding box of each label

        UnionFind uf;
        int label = 0;
        lImage.init(bImage.W, bImage.H, -1);
        
        #if __DEBUG_Progress>0
        cout<<"scan thru image .."<<bImage.W<<"x"<<bImage.H<<endl;
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

    void filterDigitBBoxes() {
        int leftEnd = lImage.W*marginFrac;
        int rightEnd = lImage.W - lImage.W*marginFrac;
        int topEnd = lImage.H*marginFrac;
        int botEnd = lImage.H - lImage.H*marginFrac;
        float wThresh = lImage.W*minWFrac;
        float hThresh = lImage.H*minHFrac;

        vector<int> labelsToRemove;
        for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) {
            int label = it->first;
            int W = it->second.W();
            int H = it->second.H();
            float AR = H*1.0/W;
            bool wOk = W>wThresh;
            bool hOk = H>hThresh;
            bool arOk = AR>minARFrac;
            bool marginOk = true;
            if((it->second.x1<leftEnd) || (it->second.x2>rightEnd)) marginOk = false;
            if((it->second.y1<topEnd) || (it->second.y2>botEnd)) marginOk = false;

            bool allOk = wOk && hOk && arOk && marginOk;
            if(!allOk) {
                #if __DEBUG_Progress>0
                printf("mark label %d to remove: W:%d, H:%d, AR:%d, Margin:%d\n",label,wOk,hOk,arOk,marginOk);
                #endif
                labelsToRemove.push_back(label);
            }
        }

        for(int n=0; n<labelsToRemove.size(); n++) {
            int label = labelsToRemove[n];
            #if __DEBUG_Progress>0
                printf("remove label %d\n",label);
            #endif
            labelBBox.erase(label);
        }
        
        if(labelBBox.size()>3) {

            // retain only top3 by size
            vector<int> sizes;
            for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) {
                sizes.push_back(it->second.size());
            }
            std::sort(sizes.begin(), sizes.end(), std::greater<int>());
            int szThresh = sizes[2];
            labelsToRemove.clear();
            for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) {
                int label = it->first;
                if(it->second.size()<szThresh) {
                    #if __DEBUG_Progress>0
                        printf("mark label %d to remove by size %d < %d\n",label,it->second.size(),szThresh);
                    #endif
                    labelsToRemove.push_back(label);
                }
            }

            for(int n=0; n<labelsToRemove.size(); n++) {
                int label = labelsToRemove[n];
                #if __DEBUG_Progress>0
                    printf("remove label %d\n",label);
                #endif
                labelBBox.erase(label);
            }
        }

        #if __DEBUG_Progress>0
        printf("Final #labels=%d\n",labelBBox.size());
        for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) {
            printf("labe=%d, bbox:",it->first);
            it->second.print(stdout);
            printf("\n");
        }
        #endif
    }

    void extractImages(Image &origImage, int padxy=2, int nW=28, int nH=28) {
        // extract image of nWxnH size with padding of padxy for each labelBBox entry
        // output:
        //  1. this->images 

        this->nW = nW;
        this->nH = nH;

        int wPad = (int)padxy * nW*1.0/28;
        int hPad = (int)padxy * nH*1.0/28;

        // arrange bbox by x-position (left to right)
        vector<pair<int,int>> labelLeftPos;
        for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) {
            labelLeftPos.push_back(pair<int,int>(it->first,it->second.x1));
        }
        std::sort(labelLeftPos.begin(), labelLeftPos.end(), pairCompByValue);
        map<int,int> label2index;
        for(int n=0; n<labelLeftPos.size(); n++) label2index[labelLeftPos[n].first] = n;
        
        #if __DEBUG_Progress>0
        printf("initialize images for each label\n");
        #endif
        for(map<int,BBox>::iterator it=labelBBox.begin(); it!=labelBBox.end(); ++it) {
            int label = it->first;
            if(label==-1) continue;
            int W = it->second.W() + wPad*2;
            int H = it->second.H() + hPad*2;
            int index = label2index[label];
            images[index].init(W,H,0);
            #if __DEBUG_Progress>0
            cout<<"label:"<<label<<", "<<W<<", "<<H<<", index="<<index<<endl;
            #endif
        }

        #if __DEBUG_Progress>0
        printf("fill image for each label\n");
        #endif
        for(int r=0; r<lImage.H; r++) {
            for(int c=0; c<lImage.W; c++) {
                int label = lImage.index(c,r);
                if(label==-1) continue; // background label
                if(labelBBox.find(label)==labelBBox.end()) continue; // label has been filtered out
                int lc = c - labelBBox[label].x1 + wPad;
                int lr = r - labelBBox[label].y1 + hPad;
                int index = label2index[label];
                images[index].set(lc,lr, 255-origImage.index(c,r));
            }
        }

        #if __DEBUG_Progress>0
        printf("resize each label image\n");
        #endif
        for(map<int,Image>::iterator it=images.begin(); it!=images.end(); ++it) {
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
        /*
        FILE *fp = fopen("tests/test_flatten.out","w");
        int noff = 0;
        for(int nimg=0;nimg<images.size();nimg++) {
            for(int n=0;n<nW*nH*1.5;n++) fprintf(fp,"%d,",flatData[noff++]);
        }
        fclose(fp);
        */
    }
};


int getSpeedValue(float *probas, int nPreds) {
    const int nClasses = 11;
    vector<int> firsts;
    vector<int> seconds;

    int offset = 0;
    bool isValidNo = false;
    for(int np=0; np<nPreds; np++) {
        vector<pair<int,float>> indexProba;
        for(int n=0;n<nClasses;n++) {
            indexProba.push_back(pair<int,float>(n, probas[n+offset]));
        }
        offset += nClasses;
        std::sort(indexProba.begin(), indexProba.end(), pairCompByValueFloat);

        int val = indexProba[nClasses-1].first;
        if(val<10) isValidNo = true; // if any of the entries has digit, isValidNo = true
        firsts.push_back(indexProba[nClasses-1].first);
        seconds.push_back(indexProba[nClasses-2].first); // note down 2nd choice to override when isValidNo=true
    }

    int speedValue = -1;
    if(isValidNo) {
        speedValue = 0;
        int mult = 1;
        for(int n=nPreds-1; n>=0; n--) {
            int dval = firsts[n];
            if(dval==10) dval = seconds[n]; // forcibly convert non-digit to digit (by 2nd best choice)
            speedValue += dval*mult;
            mult *= 10;
        }
    }
    return speedValue;
}

// Exampel JNI function calls

#define nv21dtype int
void exampleJniFunction(nv21dtype *imageArray, int W, int H, vector<int> &flattenImages, int &nImages, int &oW, int &oH) {
    Image image(W,H,imageArray); // create Image object
    Image origImage(W,H,imageArray); // create Image object
    SpeedSignExtract sobj; // initialize extraction object

    int kSz = (int)(29.0*H/100); // kernel size for adaptive thresholding
    image.adaptiveGaussianThreshold(kSz, -7); // binarize the image
    sobj.connectedComponents(image, 127); // apply connected components
    sobj.filterDigitBBoxes();
    sobj.extractImages(origImage, 0, 28, 28); // extract invidual images
    
    sobj.flatten(flattenImages, nImages, oW, oH); // get flattened YUV images
}

void exampleJniSpeedValue(float *probas, int nPreds, int &speedValue) {
    speedValue = getSpeedValue(probas, nPreds);
}

int main(int argc, char *argv[]) {

    printf("Usage: %s <mode> <input .dat> \n",argv[0]);
    printf("    mode: 0 for general processing, 1 for JNI simulation, 2 for speedValue verification\n");
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

    if (mode=='2') {
        FILE *fp = fopen(argv[2],"r");
        int nPreds, nClasses;
        fscanf(fp,"%d,%d",&nPreds,&nClasses);
        printf("nPreds=%d, nClasses=%d\n",nPreds,nClasses);
        float *probas = new float[nPreds*nClasses];
        for(int n=0;n<nPreds*nClasses; n++) fscanf(fp,",%e",&probas[n]);
        fclose(fp);

        int speedValue;
        exampleJniSpeedValue(probas, nPreds, speedValue);
        printf("\nSpeed Value = %d\n\n", speedValue);

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

    image.adaptiveGaussianThreshold(11,5);
    image.dump2File("tests/adaptfilt_image.out");
    sobj.connectedComponents(image, 127);
    sobj.extractImages(image);
    
    cout<<"done.";
    return 0;
}