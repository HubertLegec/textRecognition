#include "DetectedWord.h"

const string &DetectedWord::getText() const {
    return text;
}

float DetectedWord::getConfidence() const {
    return confidence;
}

DetectedWord::DetectedWord(string text, float confidence) {
    this->text = text;
    this->confidence = confidence;
}

string DetectedWord::toString() const {
    return "Word '" + text + "' with confidence: " + to_string(confidence);
}
