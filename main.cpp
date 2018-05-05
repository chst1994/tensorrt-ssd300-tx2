#include <iostream>
#include <algorithm>
#include "tensorRTplugin/tensorNet.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "util/cuda/cudaRGB.h"
#include "util/loadImage.h"

using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace cv;

const char* model  = "/home/nvidia/tensorrt-ssd300-tx2-15fps/deploy.prototxt";
const char* weight = "/home/nvidia/tensorrt-ssd300-tx2-15fps/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
const char* label  = "/home/nvidia/caffe-ssd/caffe/data/car/labelmap_voc.prototxt";

static const uint32_t BATCH_SIZE = 1;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT1 = "detection_out";
const char* OUTPUT2 = "";
const char* OUTPUT3 = "";
const char* OUTPUT_BLOB_NAME = "detection_out";

/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}

cudaError_t cudaPreImageNetMean( float3* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value);

int main()
{ 
    TensorNet tensorNet;
    tensorNet.caffeToTRTModel( model, weight, std::vector<std::string>{ OUTPUT1 }, BATCH_SIZE);
    tensorNet.createInference();

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);
    DimsCHW dims1    = tensorNet.getTensorDims(OUTPUT1);

    cout << "INPUT Tensor Shape is: C: "  <<dimsData.c()<< "  H: "<<dimsData.h()<<"  W:  "<<dimsData.w()<<endl;
    cout << "OUTPUT1 Tensor Shape is: C: "<<dims1.c()<<"  H: "<<dims1.h()<<"  W: "<<dims1.w()<<endl;
    cout << "OUTPUT Tensor Shape is: C: "<<dimsOut.c()<<"  H: "<<dimsOut.h()<<"  W: "<<dimsOut.w()<<endl;
    float* data    = allocateMemory( dimsData , (char*)"input blob");
    float* output  = allocateMemory( dimsOut  , (char*)"output blob");
    float* output1 = allocateMemory( dims1    , (char*)"output1");

    int height = 300;
    int width  = 300;

    Mat frame;
    Mat frame_float;

    frame = cv::imread("/home/nvidia/tensorrt-ssd300-tx2-15fps/000001.jpg", IMREAD_COLOR);
    resize(frame, frame, Size(300,300));
    void* imgCPU;
    void* imgCUDA;
    float * cst;
    float *p;
    p = (float *) malloc (sizeof(float)*3*300*300);
    cudaMalloc((void**)&cst, 300*300 * 3 * sizeof(float));
    const uint32_t imgWidth  = 300;
    const uint32_t imgHeight = 300;
    const uint32_t imgPixels = imgWidth * imgHeight;
    const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 3;





    const size_t size = width * height * sizeof(float3);
    if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
        {
            cout <<"Cuda Memory allocation error occured."<<endl;
            return false;
        }
    for (int i=0; i<100; i++ )
    {   float start=getTickCount();
        frame.convertTo(frame, CV_32F, 1, 0);
/*
        
        */
        if( !loadImageBGR( frame , (float3**)&imgCPU, (float3**)&imgCUDA, &height, &width))
        {
            printf("failed to load image '%s'\n", "Image");
            return 0;
        }
        memcpy(imgCPU,frame.data,imgSize);
        if( CUDA_FAILED(cudaPreImageNetMean( (float3*)imgCPU, width, height, cst, dimsData.w(), dimsData.h(), make_float3(123.0f,117.0f,104.0f))))
        {
            cout <<"Cuda pre image net mean failed. " <<endl;
            return 0;
        }
        //cudaMemcpy(p,cst,imgWidth * imgHeight * sizeof(float) * 3,cudaMemcpyDeviceToHost);
       // float start1=getTickCount();
        void* buffers[] = { cst, output1};

        tensorNet.imageInference( buffers, 2, BATCH_SIZE);
        vector<vector<float> > detections;

        for (int k=0;k<10;k++)
        {
            if(output1[k*7+2]>0.1){
               // int xi=output1[7*k+3];
               // int yi=output1[7*k+4];
               // int xa=output1[7*k+5];
               // int ya=output1[7*k+6];
              //  rectangle(frame,Rect(xi,yi,xa,ya),Scalar(0,255,255),1,1,0);
                cout<<(output1[7*k+0])<<"  "<<(output1[7*k+1])<<"  "<<(output1[7*k+2])<<"  "<<(output1[7*k+3])<<"   "<<(output1[7*k+4])<<"   "<<(output1[7*k+5])<<"   "<<(output1[7*k+6])<<endl;
            }
        }

       // imshow("Objects Detected", frame);
        //waitKey(1);
        float end2=getTickCount();
        //double time2 = (start1-start)/getTickFrequency(); 
        double time1 = (end2-start)/getTickFrequency(); 
        cout<<"time1 ="<<time1<<endl; 
        //cout<<"time2 ="<<time2<<endl; 
    }
   
   CUDA(cudaFreeHost(imgCPU));
    tensorNet.destroy();

    return 0;

}
