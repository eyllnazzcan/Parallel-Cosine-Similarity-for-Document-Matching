#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <cctype>
using namespace std;

vector<string> split_by_space(const string& text) {
    vector<string> tokens;
    istringstream iss(text);
    string word;
    while (iss >> word) tokens.push_back(word);
    return tokens;
}

struct Document {
    string raw_text;
    vector<string> tokens;
    unordered_map<string, double> tf;
    unordered_map<string, double> tfidf;
};

double cosine_similarity(const unordered_map<string, double>& vec1,
                         const unordered_map<string, double>& vec2) {
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (const auto& pair : vec1) {
        norm1 += pair.second * pair.second;
        if (vec2.count(pair.first)) dot += pair.second * vec2.at(pair.first);
    }
    for (const auto& pair : vec2) norm2 += pair.second * pair.second;
    if (norm1 == 0 || norm2 == 0) return 0.0;
    return dot / (sqrt(norm1) * sqrt(norm2));
}

void run_parallel_processing(int rank, int size) {
    double start_time = MPI_Wtime();
    vector<string> my_lines;

    // Rank 0 reads and distributes the file
    if (rank == 0) {
        ifstream file("AllCombined_cleaned.txt");
        if (!file.is_open()) {
            cerr << "Failed to open dataset.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line;
        vector<vector<string>> chunks(size);
        int idx = 0, total_lines_read = 0, max_docs = 100000;
        while (getline(file, line)) {
            if (line.empty()) continue;
            if (++total_lines_read > max_docs) break;
            chunks[idx % size].push_back(line);
            idx++;
        }
        file.close();

        for (int dest = 1; dest < size; ++dest) {
            int count = chunks[dest].size();
            MPI_Send(&count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            for (const string& l : chunks[dest]) {
                int len = l.length();
                MPI_Send(&len, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Send(l.c_str(), len, MPI_CHAR, dest, 0, MPI_COMM_WORLD);
            }
        }
        my_lines = chunks[0];
    } else {
        int count;
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; ++i) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            vector<char> buf(len);
            MPI_Recv(buf.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            my_lines.emplace_back(buf.begin(), buf.end());
        }
    }

    vector<Document> docs;
    unordered_map<string, int> local_df;
    set<string> local_vocab;

    for (const string& line : my_lines) {
        Document doc;
        doc.raw_text = line;
        doc.tokens = split_by_space(line);
        unordered_set<string> seen;
        for (const string& w : doc.tokens) {
            doc.tf[w]++;
            seen.insert(w);
            local_vocab.insert(w);
        }
        for (const auto& w : seen) local_df[w]++;
        docs.push_back(doc);
    }

    // Global vocab gathering and broadcasting
    vector<string> global_vocab;
    if (rank == 0) {
        set<string> merged_vocab(local_vocab.begin(), local_vocab.end());
        for (int i = 1; i < size; ++i) {
            int count;
            MPI_Recv(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < count; ++j) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string word(len, ' ');
                MPI_Recv(&word[0], len, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                merged_vocab.insert(word);
            }
        }
        global_vocab.assign(merged_vocab.begin(), merged_vocab.end());
    } else {
        int count = local_vocab.size();
        MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        for (const auto& word : local_vocab) {
            int len = word.size();
            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(word.c_str(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    int vocab_size;
    if (rank == 0) vocab_size = global_vocab.size();
    MPI_Bcast(&vocab_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) global_vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        int len;
        if (rank == 0) len = global_vocab[i].size();
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) global_vocab[i].resize(len);
        MPI_Bcast(&global_vocab[i][0], len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    unordered_map<string, int> word_to_index;
    for (int i = 0; i < global_vocab.size(); ++i)
        word_to_index[global_vocab[i]] = i;

    vector<int> local_df_vec(vocab_size, 0);
    for (const auto& pair : local_df)
        local_df_vec[word_to_index[pair.first]] = pair.second;

    vector<int> global_df_vec(vocab_size, 0);
    MPI_Allreduce(local_df_vec.data(), global_df_vec.data(), vocab_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int local_doc_count = docs.size(), total_docs = 0;
    MPI_Allreduce(&local_doc_count, &total_docs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    for (auto& doc : docs) {
        for (auto& p : doc.tf) p.second /= doc.tokens.size();
        for (const auto& p : doc.tf) {
            double idf = log(1.0 + (double)total_docs / (1.0 + global_df_vec[word_to_index[p.first]]));
            doc.tfidf[p.first] = p.second * idf;
        }
    }

    // Distributed pairwise similarity
    vector<pair<int, int>> pairs;
    int global_index = 0;
    for (int i = 0; i < total_docs; ++i) {
        for (int j = i + 1; j < total_docs; ++j) {
            if (global_index % size == rank) {
                pairs.emplace_back(i, j);
            }
            global_index++;
        }
    }

    vector<Document> all_docs;
    for (const auto& doc : docs) all_docs.push_back(doc);
    vector<Document> gathered_docs;
    int my_count = docs.size();
    vector<int> all_counts(size);
    MPI_Allgather(&my_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + all_counts[i - 1];
    int total = displs[size - 1] + all_counts[size - 1];
    vector<string> all_texts(total);
    for (int i = 0; i < docs.size(); ++i) all_texts[displs[rank] + i] = docs[i].raw_text;
    for (int i = 0; i < total; ++i) {
        if (i < displs[rank] || i >= displs[rank] + my_count) {
            Document doc;
            doc.raw_text = all_texts[i];
            doc.tokens = split_by_space(doc.raw_text);
            for (const auto& w : doc.tokens) doc.tf[w]++;
            for (auto& p : doc.tf) p.second /= doc.tokens.size();
            for (const auto& p : doc.tf) {
                double idf = log(1.0 + (double)total_docs / (1.0 + global_df_vec[word_to_index[p.first]]));
                doc.tfidf[p.first] = p.second * idf;
            }
            all_docs.push_back(doc);
        }
    }

    ofstream out("similarity_rank_" + to_string(rank) + ".txt");
    for (const auto& pr : pairs) {
        double sim = cosine_similarity(all_docs[pr.first].tfidf, all_docs[pr.second].tfidf);
        out << "Doc " << pr.first << " vs Doc " << pr.second << ": " << sim << "\n";
    }
    out.close();

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    cout << "Rank " << rank << " execution time: " << elapsed << " seconds." << endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    run_parallel_processing(rank, size);
    MPI_Finalize();
    return 0;
}