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

Mat custom_blur(Mat input_mat, Mat kernel) {
    Mat output_mat;
    vector<Vec3b> filter;

//    cout << input_mat.cols << ' ' << input_mat.rows << endl;

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

            if(i == 1 && j == 1) {
                cout << min_mat.cols << ' ' << min_mat.rows << endl << min_mat << endl << input_mat(zone) << endl << mul_mat << endl;
                cout << point << endl;
            }
        }
    }

    output_mat = Mat(input_mat.rows-2, input_mat.cols-2, input_mat.type(), &filter[0]);

    cout << input_mat.cols << ' ' << input_mat.rows << endl << output_mat.cols << ' ' << output_mat.rows << endl;
    imshow("Custom blur", output_mat);
    return output_mat;
}

Mat custom_gradient(Mat GRAY_mat, Mat kernel) {
    Mat output_mat;
    vector<uchar> filter;

//    cout << input_mat.cols << ' ' << input_mat.rows << endl;

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

            if(i == 1 && j == 1) {
                cout << min_mat.cols << ' ' << min_mat.rows << endl << min_mat << endl << GRAY_mat(zone) << endl << mul_mat << endl;
                cout << point << endl;
            }
        }
    }

    output_mat = Mat(GRAY_mat.rows-2, GRAY_mat.cols-2, GRAY_mat.type(), &filter[0]);

    cout << GRAY_mat.cols << ' ' << GRAY_mat.rows << endl << output_mat.cols << ' ' << output_mat.rows << endl;
    imshow("Custom gradient", output_mat);
    return output_mat;
}

int main(int argc, char* argv[])
{
    Mat image = imread(argv[1]);
    Mat image_blur = imread(argv[1]);
    Mat imageGRAY;

    int width_of_kernel = 3;
    int height_of_kernel = 3; /*cout << height << ' ' << width << endl;*/
    int sigma = 3;
    int ksize = (sigma*5);/*|1;*/

    cvtColor(image, imageGRAY, COLOR_BGR2GRAY);
    GaussianBlur(image_blur, image_blur, Size(ksize, ksize), sigma, sigma);

//    namedWindow("Original image");
//    namedWindow("Blur Gaussian");
//    namedWindow("Custom blur");



    float matrix[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    Mat kernel(3, 3, CV_32F, matrix);
    kernel = kernel/9;
//    cout << kernel << endl;
    float matrix_grad[] = {1,0,-1,2,0,-2,1,0,-1};
    Mat kernel_grad(3, 3, CV_32F, matrix_grad);
    kernel_grad = kernel_grad/9;

    Mat custom_image = custom_blur(image, kernel);
    Mat custom_blur_image = custom_gradient(imageGRAY, kernel_grad);

    imshow("Original image", image);
    imshow("Blur Gaussian", image_blur);
//    imshow("Custom blur", custom_image);

    waitKey(0);

    return 0;
}



//    cout << image.cols << ' ' << image.rows << endl;
//    for(int i = 1; i < image.rows - 1; i++) {
//        for(int j = 1; j < image.cols - 1; j++) {
////            Mat m_min(image, Rect (j-1, i-1, 3, 3));
//            Rect zone(j-1, i-1, 3, 3);
//            Mat m_min = image(zone).clone();
////            cout << m_min << endl;
//            Mat mult_mat(3, 3, m_min.type());
//            for(int n=0; n < m_min.cols; n++) {
//                for(int m=0; m < m_min.cols; m++) {
//                    mult_mat.at<Vec3d>(n,m) = m_min.at<Vec3d>(n,m)*kernel.at<float>(n,m);
//                    if( i == 1 && j == 1) {
////                        cout << m_min.at<Vec3d>(n,m) << endl;/* << kernel.at<float>(n,m) << endl << mult_mat.at<Vec3d>(n,m) << endl;*/
//                        if(n == 0 && m == 0) {
//                            cout << m_min << endl;
//                        }
//                        cout << m_min.at<Vec3d>(n,m);
//                    }
//                }
//                if(i == 1 && j == 1)
//                    cout << endl;
//            }
////            mult_mat = m_min/9;
////            cout << mult_mat << endl;
//            if(i == 1 && j == 1) {
////                Mat exp = Mat (3, 3, mult_mat.type());
////                exp = m_min*kernel;
//                cout << m_min << endl;
////                cout << kernel << endl;
////                cout << mult_mat << endl;
////                cout << exp << endl;
//            }
//            Vec3d point( sum(mult_mat)[0], sum(mult_mat)[1], sum(mult_mat)[2]);
//            filter.push_back(point);
//        }
//    }

//    Mat custom_image(image.rows-2, image.cols-2, image.type(), &filter[0]);
