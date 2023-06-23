#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <opencv2/opencv.hpp>

const std::string weightsFile = "yolov3-tiny.weights";
const std::string configFile = "yolov3-tiny.cfg";
const std::string classNamesFile = "coco.names";

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& detections, const std::vector<cv::Mat>& boxes,
    const std::vector<float>& confidences, const std::vector<int>& classIds)
{
    for (size_t i = 0; i < detections.size(); ++i)
    {
        cv::Mat detection = detections[i];
        cv::Mat box = boxes[i];
        float confidence = confidences[i];
        int classId = classIds[i];

        if (confidence > 0.5)
        {
            int left = static_cast<int>(box.at<float>(0, 0) * frame.cols);
            int top = static_cast<int>(box.at<float>(0, 1) * frame.rows);
            int right = static_cast<int>(box.at<float>(0, 2) * frame.cols);
            int bottom = static_cast<int>(box.at<float>(0, 3) * frame.rows);

            cv::Rect roi(left, top, right - left, bottom - top);
            cv::Mat cropped = frame(roi);

            // Save cropped image
            std::stringstream ss;
            ss << "detection_" << std::setfill('0') << std::setw(4) << i << ".jpg";
            cv::imwrite(ss.str(), cropped);

            // Save detection details to CSV file
            std::ofstream outfile("detections.csv", std::ios_base::app);
            outfile << std::time(nullptr) << "," << classNamesFile[classId] << "," << confidence << ","
                << left << "," << top << "," << right << "," << bottom << std::endl;
        }
    }
}

int main()
{
    // Load YOLOv3-tiny
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(configFile, weightsFile);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class names
    std::vector<std::string> classNames;
    std::ifstream classNamesFileIn(classNamesFile);
    std::string className;
    while (std::getline(classNamesFileIn, className))
    {
        classNames.push_back(className);
    }

    // Open video capture
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera!" << std::endl;
        return -1;
    }

    // Create window
    cv::namedWindow("YOLOv3-tiny Object Detection", cv::WINDOW_NORMAL);

    while (cv::waitKey(1) < 0)
    {
        // Read frame from camera
        cv::Mat frame;
        cap >> frame;

        // Create a blob from the frame
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);

        // Set the blob as input to the network
        net.setInput(blob);

        // Forward pass to get output
        std::vector<cv::String> outNames = { "yolo_16", "yolo_23" };
        std::vector<cv::Mat> outs;
        net.forward(outs, outNames);

        // Post-process the output
        std::vector<cv::Mat> detections;
        std::vector<cv::Mat> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;
        for (const auto& out : outs)
        {
            float* data = reinterpret_cast<float*>(out.data);
            for (int j = 0; j < out.rows; ++j, data += out.cols)
            {
                cv::Mat scores = out.row(j).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

                if (confidence > 0.5)
                {
                    int centerX = static_cast<int>(data[0] * frame.cols);
                    int centerY = static_cast<int>(data[1] * frame.rows);
                    int width = static_cast<int>(data[2] * frame.cols);
                    int height = static_cast<int>(data[3] * frame.rows);

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    detections.push_back(out);
                    boxes.push_back(cv::Mat(cv::Rect(left, top, width, height)));
                    confidences.push_back(static_cast<float>(confidence));
                    classIds.push_back(classIdPoint.x);
                }
            }
        }

        // Perform post-processing and save detections
        postprocess(frame, detections, boxes, confidences, classIds);

        // Draw bounding boxes and class labels on the frame
        for (size_t i = 0; i < detections.size(); ++i)
        {
            cv::Mat detection = detections[i];
            cv::Mat box = boxes[i];
            float confidence = confidences[i];
            int classId = classIds[i];

            int left = static_cast<int>(box.at<float>(0, 0) * frame.cols);
            int top = static_cast<int>(box.at<float>(0, 1) * frame.rows);
            int right = static_cast<int>(box.at<float>(0, 2) * frame.cols);
            int bottom = static_cast<int>(box.at<float>(0, 3) * frame.rows);

            cv::rectangle(frame, cv::Rect(left, top, right - left, bottom - top), cv::Scalar(0, 255, 0), 2);

            std::stringstream ss;
            ss << classNames[classId] << " (" << std::fixed << std::setprecision(2) << confidence << ")";
            cv::putText(frame, ss.str(), cv::Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // Show the frame
        cv::imshow("YOLOv3-tiny Object Detection", frame);
    }

    // Release the capture and destroy windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

