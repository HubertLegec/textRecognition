#include <string>
#include <iostream>
#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "TextRecognizer.h"
#include "RecognitionModule.h"

using namespace std;

int main(int argc, char *argv[]) {
    string classifierNM1Path = "trained_classifierNM1.xml";
    string classifierNM2Path = "trained_classifierNM2.xml";

    cout << "Podaj sciezke do pliku z obrazem:" << endl;
    string imagePath;
    cin >> imagePath;

    RecognitionModule recognitionModule(imagePath);
    Mat outImg = recognitionModule.process(classifierNM1Path, classifierNM2Path);
    auto decompositions = recognitionModule.getDecompositions();
    auto words = recognitionModule.getWords();

    cout << "--- words ---" << endl;
    for (auto word : words) {
        cout << word.toString() << endl;
    }
    cout << "-------------" << endl;

    for(int i = 0; i < decompositions.size(); i++) {
        namedWindow("decomposition" + to_string(i), WINDOW_NORMAL);
        imshow("decomposition" + to_string(i), decompositions[i]);
    }
    namedWindow("recognition", WINDOW_NORMAL);
    imshow("recognition", outImg);

    waitKey(0);
    return 0;
}