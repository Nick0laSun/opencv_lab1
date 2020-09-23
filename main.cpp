#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

Mat custom_blur(Mat input_mat, Mat kernel) {
    Mat output_mat;
    vector<Vec3b> filter;

    for(int i = 1; i < input_mat.rows-1; i++) {
        for(int j = 1; j < input_mat.cols-1; j++) {
            Rect zone(j-1, i-1, 3, 3);
            Mat min_mat = input_mat(zone).clone();
            Mat mul_mat(3, 3, min_mat.type());

            for(int n = 0; n < min_mat.cols; n++) {
                for(int m = 0; m < min_mat.cols; m++) {
                    mul_mat.at<Vec3b>(n,m) = min_mat.at<Vec3b>(n,m)*kernel.at<float>(n,m);
                }
            }

            Vec3b point(sum(mul_mat)[0], sum(mul_mat)[1], sum(mul_mat)[2]);
            filter.push_back(point);
        }
    }

    output_mat = Mat(input_mat.rows-2, input_mat.cols-2, input_mat.type(), &filter[0]);

    imshow("Custom blur", output_mat);
    return output_mat;
}

Mat custom_gradient(Mat GRAY_mat, Mat kernel) {
    Mat output_mat;
    vector<uchar> filter;


    for(int i = 1; i < GRAY_mat.rows-1; i++) {
        for(int j = 1; j < GRAY_mat.cols-1; j++) {
            Rect zone(j-1, i-1, 3, 3);
            Mat min_mat = GRAY_mat(zone).clone();
            Mat mul_mat(3, 3, min_mat.type());

            for(int n = 0; n < min_mat.cols; n++) {
                for(int m = 0; m < min_mat.cols; m++) {
                    mul_mat.at<uchar>(n,m) = min_mat.at<uchar>(n,m)*kernel.at<float>(n,m);
                }
            }

            uchar point(sum(mul_mat)[0]);
            filter.push_back(point);

        }
    }

    output_mat = Mat(GRAY_mat.rows-2, GRAY_mat.cols-2, GRAY_mat.type(), &filter[0]);

    imshow("Custom gradient", output_mat);
    return output_mat;
}

int main(int argc, char* argv[])
{
    Mat image = imread(argv[1]);
    Mat image_blur = imread(argv[1]);
    Mat imageGRAY;

    int sigma = 3;
    int ksize = (sigma*5);

    cvtColor(image, imageGRAY, COLOR_BGR2GRAY);
    GaussianBlur(image_blur, image_blur, Size(ksize, ksize), sigma, sigma);

    float matrix[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    Mat kernel(3, 3, CV_32F, matrix);
    kernel = kernel/9;

    float matrix_grad[] = {1,0,-1,2,0,-2,1,0,-1};
    Mat kernel_grad(3, 3, CV_32F, matrix_grad);
    kernel_grad = kernel_grad/9;

    custom_blur(image, kernel);
    custom_gradient(imageGRAY, kernel_grad);

    imshow("Original image", image);
    imshow("Blur Gaussian", image_blur);

    waitKey(0);

    return 0;
}
