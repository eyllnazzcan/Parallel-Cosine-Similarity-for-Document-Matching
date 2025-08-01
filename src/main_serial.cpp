#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <chrono>

using namespace std;

vector<string> split_by_space(const string& text) {
    vector<string> tokens;
    istringstream iss(text);
    string word;
    while (iss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

struct Document {
    string text;
    vector<string> tokens;
    unordered_map<string, double> tf;
    unordered_map<string, double> tfidf;
};

double cosine_similarity(const unordered_map<string, double>& vec1,
                         const unordered_map<string, double>& vec2) {
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;

    for (const auto& pair : vec1) {
        norm1 += pair.second * pair.second;
        if (vec2.count(pair.first))
            dot += pair.second * vec2.at(pair.first);
    }

    for (const auto& pair : vec2)
        norm2 += pair.second * pair.second;

    if (norm1 == 0 || norm2 == 0) return 0.0;
    return dot / (sqrt(norm1) * sqrt(norm2));
}

int main() {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    ifstream file("AllCombined_cleaned.txt");
    ofstream output("serial_similarity_scores.txt");

    if (!file.is_open()) {
        cerr << "Failed to open input file.\n";
        return 1;
    }

    string line;
    vector<Document> docs;
    unordered_map<string, int> df;
    int total_docs = 0;

    // Read, tokenize, compute TF and document frequency
    while (getline(file, line)) {

        if (line.empty()) continue;

        Document doc;
        doc.text = line;
        doc.tokens = split_by_space(line);

        if (doc.tokens.empty()) continue;

        unordered_set<string> seen;
        for (const auto& word : doc.tokens) {
            doc.tf[word] += 1.0;
            seen.insert(word);
        }
        for (const auto& word : seen)
            df[word]++;

        docs.push_back(doc);
        total_docs++;
    }

    // Compute TF normalization and TF-IDF
    for (auto& doc : docs) {
        for (auto& pair : doc.tf)
            pair.second /= doc.tokens.size();

        for (const auto& pair : doc.tf) {
            double idf = log(1.0 + (double)total_docs / (1.0 + df[pair.first]));
            doc.tfidf[pair.first] = pair.second * idf;
        }
    }

    // Pairwise cosine similarity
    for (size_t i = 0; i < docs.size(); ++i) {
        for (size_t j = i + 1; j < docs.size(); ++j) {
            double sim = cosine_similarity(docs[i].tfidf, docs[j].tfidf);
            output << "Doc " << i << " vs Doc " << j << ": " << sim << "\n";
        }
    }

    auto end_time = high_resolution_clock::now();
    duration<double> elapsed = end_time - start_time;

    cout << "Total execution time: " << elapsed.count() << " seconds\n";

    return 0;
}
