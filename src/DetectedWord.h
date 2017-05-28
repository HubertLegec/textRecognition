#ifndef TEXTRECOGNITION_DETECTEDWORD_H
#define TEXTRECOGNITION_DETECTEDWORD_H

#include <string>

using namespace std;

class DetectedWord {
private:
    string text;
    float confidence;
    int channelNum;
public:
    DetectedWord(string text, float confidence);

    const string &getText() const;

    float getConfidence() const;

    string toString() const;
};


#endif //TEXTRECOGNITION_DETECTEDWORD_H
