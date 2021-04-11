/*
 * utility functions for visualization of results (disparity in color, warped output, ...)
 */

#pragma once
#include <sstream>
#include <fstream>

#if (CV_MAJOR_VERSION ==2)
#include <opencv2/contrib/contrib.hpp> // needed for applyColorMap!
#endif

#include "point_cloud.h"
#include "point_cloud_list.h"

/* compute gamma correction (just for display purposes to see more details in farther away areas of disparity image)
 * Input: img   - image
 *        gamma - gamma value
 * Output: gamma corrected image
 */
Mat correctGamma( Mat& img, double gamma ) {
 double inverse_gamma = 1.0 / gamma;

 Mat lut_matrix(1, 256, CV_8UC1 );
 uchar * ptr = lut_matrix.ptr();
 for( int i = 0; i < 256; i++ )
   ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

 Mat result;
 LUT( img, lut_matrix, result );

 return result;
}


static void getDisparityForDisplay(const Mat_<float> &disp, Mat &dispGray, Mat &dispColor, float numDisparities, float minDisp = 0.0f){
	float gamma = 2.0f; // to get higher contrast for lower disparity range (just for color visualization)
	disp.convertTo(dispGray,CV_16U,65535.f/(numDisparities-minDisp),-minDisp*65535.f/(numDisparities-minDisp));
	Mat disp8;
	disp.convertTo(disp8,CV_8U,255.f/(numDisparities-minDisp),-minDisp*255.f/(numDisparities-minDisp));
	if(minDisp == 0.0f)
		disp8 = correctGamma(disp8,gamma);
	applyColorMap(disp8, dispColor, COLORMAP_JET);
	for(int y = 0; y < dispColor.rows; y++){
		for(int x = 0; x < dispColor.cols; x++){
			if(disp(y,x) <= 0.0f)
				dispColor.at<Vec3b>(y,x) = Vec3b(0,0,0);
		}
	}
}

static void convertDisparityDepthImage(const Mat_<float> &dispL, Mat_<float> &d, float f, float baseline){
	d = Mat::zeros(dispL.rows, dispL.cols, CV_32F);
	for(int y = 0; y < dispL.rows; y++){
		for(int x = 0; x < dispL.cols; x++){
			d(y,x) = disparityDepthConversion(f,baseline,dispL(y,x));
		}
	}
}

static string getColorString(uint8_t color){
	stringstream ss;
	ss << (int)color << " " << (int)color << " " << (int)color;
	return ss.str();
}


static string getColorString(Vec3b color){
	stringstream ss;
	ss << (int)color(2) << " " << (int)color(1) << " " << (int)color(0);
	return ss.str();
}

static string getColorString(Vec3i color){
	stringstream ss;
	ss << (int)((float)color(2)/256.f) << " " << (int)((float)color(1)/256.f) << " " << (int)((float)color(0)/256.f);
	return ss.str();
}

static void storePlyFileBinaryPointCloud (char* plyFilePath, PointCloudList &pc, Mat_<float> &distImg) {
    cout << "store 3D points to ply file" << endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath,"wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
//    fprintf(outputPly, "property float nx\n");
//    fprintf(outputPly, "property float ny\n");
//    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    distImg = Mat::zeros(pc.rows,pc.cols,CV_32F);

    //write data
#pragma omp parallel for
    for(size_t i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
//        const float4 normal = p.normal;
        float4 X = p.coord;
        const char color_r = (int)p.texture4[2];
        const char color_g = (int)p.texture4[1];
        const char color_b = (int)p.texture4[0];
        /*const int color = 127.0f;*/
        /*printf("Writing point %f %f %f\n", X.x, X.y, X.z);*/

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            /*myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;*/
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
//            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
//            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
//            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&color_r,  sizeof(char), 1, outputPly);
            fwrite(&color_g,  sizeof(char), 1, outputPly);
            fwrite(&color_b,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

static void getNormalsForDisplay(const Mat &normals, Mat &normals_display, int rtype = CV_16U){
	if(rtype == CV_8U)
		normals.convertTo(normals_display,CV_8U,128,128);
	else
		normals.convertTo(normals_display,CV_16U,32767,32767);
	cvtColor(normals_display,normals_display,COLOR_RGB2BGR);
}
