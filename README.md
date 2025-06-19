# SpeedSignCpp

## JNI Functions:
1: JNI function to extract digit images from main image (imageArray, W, H); output in form of (flattenImages, nImages, oW, oH)
#define nv21dtype int
void exampleJniFunction(nv21dtype *imageArray, int W, int H, vector<int> &flattenImages, int &nImages, int &oW, int &oH)

2: JNI function to get speed value using classifier probabilities (probas)
void exampleJniSpeedValue(float *probas, int nPreds, int &speedValue)

## Example execution
1. Process NV21 image to get separated digit images, each should be used with classifier
./a.exe 1 tests/exInputs/ssimg_1931_120.dat --> create flatImages.dat

2. Run relevant cells in verify.ipynb --> creates probas.dat

3. Process classifer probabilities (in probas.out) to get speed value
./a.exe 2 tests/probas.dat --> speed value printed out

## Test cases
1. tests/ssimg_1235_100.dat --> flatImages_1235.out, probas_1235_100.dat    
2. tests/ssimg_1931_120.dat --> flatImages_1931.out, probas_1931_120.dat
3. tests/ssimg_1987_60.dat  --> flatImages_1987.out, probas_1987_60.dat
