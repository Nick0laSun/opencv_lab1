#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

//vector<vector<double>> getgaussian(int height, int width) {

//    vector<vector<double>> matrix;
//    double sum = 0.0;

//    for(int i = 0; i < height; i++) {
//        vector<double> line;
//        for (int j = 0; j < width; j++) {
//            line.push_back(1);
//            sum += line[j];
//        }
//        matrix.push_back(line);
//    }

//    for(int i = 0; i < height; i++) {
//        for (int j = 0; j < width; j++) {
//            matrix[i][j] /= sum;
//        }
//    }

//    return  matrix;
//};

int main(int argc, char* argv[])
{
    Mat image = imread(argv[1]);
    Mat image_blur = imread(argv[1]);

    int width_of_kernel = 3;
    int height_of_kernel = 3; /*cout << height << ' ' << width << endl;*/
    int sigma = 3;
    int ksize = (sigma*5);/*|1;*/

    GaussianBlur(image_blur, image_blur, Size(ksize, ksize), sigma, sigma);

    namedWindow("Original image");
    namedWindow("Blur Gaussian");
    namedWindow("Custom blur");


    vector<Vec3d> filter;
    float matrix[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    Mat kernel(3, 3, CV_32F, matrix);
    kernel = kernel/9;
//    cout << kernel << endl;


    for(int i = 1; i < image.cols - 1; i++) {
        for(int j = 1; j < image.rows - 1; j++) {
            Mat m_min(image, Rect (i-1, j-1, 3, 3));
            Mat mult_mat(3, 3, m_min.type());
            for(int n=0; n < m_min.cols; n++) {
                for(int m=0; m < m_min.rows; m++) {
                    mult_mat.at<Vec3d>(n,m) = m_min.at<Vec3d>(n,m)*kernel.at<float>(n,m);
                }
            }
            Vec3d point( sum(mult_mat)[0], sum(mult_mat)[1], sum(mult_mat)[2]);
            filter.push_back(point);
        }
    }

    Mat custom_image(image.rows-2, image.cols-2, image.type(), &filter[0]);

    imshow("Original image", image);
    imshow("Blur Gaussian", image_blur);
    imshow("Custom blur", custom_image);

    waitKey(0);

    return 0;
}


