#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    out += "'";
    return out;
}

static std::vector<std::string> load_samples(const fs::path& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open sample-order file.");

    std::vector<std::string> samples;
    std::string line;

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (!line.empty())
            samples.push_back(line);
    }

    return samples;
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            std::cerr
                << "Usage:\n"
                << argv[0]
                << " INPUT.pyseer.gz SAMPLE_ORDER.txt"
                << " OUT.bin OUT.tsv MAX_FEATURES MSC\n";
            return 2;
        }

        const fs::path input_gz   = argv[1];
        const fs::path sample_txt = argv[2];
        const fs::path out_bin    = argv[3];
        const fs::path out_tsv    = argv[4];

        const std::uint64_t max_features =
            std::stoull(argv[5]);

        const std::uint64_t msc_threshold =
            std::stoull(argv[6]);

        if (!fs::exists(input_gz))
            throw std::runtime_error("Input gzip does not exist.");

        if (!fs::exists(sample_txt))
            throw std::runtime_error("Sample-order file does not exist.");

        auto samples = load_samples(sample_txt);

        const std::size_t N = samples.size();

        if (N != 4227)
            throw std::runtime_error(
                "Expected 4227 samples, observed "
                + std::to_string(N)
            );

        const std::size_t bytes_per_feature =
            (N + 7) / 8;

        if (bytes_per_feature != 529)
            throw std::runtime_error(
                "Unexpected bytes-per-feature."
            );

        // Build zero-allocation lookup keys.
        std::unordered_map<std::string_view, std::uint16_t> sample_to_idx;
        sample_to_idx.reserve(N * 2);

        for (std::size_t i = 0; i < N; ++i) {
            auto ok = sample_to_idx.emplace(
                std::string_view(samples[i]),
                static_cast<std::uint16_t>(i)
            );

            if (!ok.second)
                throw std::runtime_error(
                    "Duplicate sample in sample-order file: "
                    + samples[i]
                );
        }

        fs::create_directories(out_bin.parent_path());
        fs::create_directories(out_tsv.parent_path());

        const fs::path bin_partial =
            out_bin.string() + ".partial";

        const fs::path tsv_partial =
            out_tsv.string() + ".partial";

        // Benchmark intentionally starts clean.
        if (fs::exists(bin_partial))
            fs::remove(bin_partial);

        if (fs::exists(tsv_partial))
            fs::remove(tsv_partial);

        std::ofstream bout(
            bin_partial,
            std::ios::binary | std::ios::trunc
        );

        std::ofstream tout(
            tsv_partial,
            std::ios::out | std::ios::trunc
        );

        if (!bout || !tout)
            throw std::runtime_error(
                "Cannot create benchmark output files."
            );

        tout
            << "Feature Index\tRaw Record\tSequence\t"
            << "Prevalence\tMinor State Count\tByte Offset\n";

        std::string cmd =
            "gzip -dc -- "
            + shell_quote(input_gz.string());

        FILE* pipe = popen(cmd.c_str(), "r");

        if (!pipe)
            throw std::runtime_error(
                "Failed to launch gzip reader."
            );

        char* line_ptr = nullptr;
        std::size_t line_capacity = 0;

        std::vector<unsigned char> packed(
            bytes_per_feature,
            0
        );

        std::uint64_t raw_records = 0;
        std::uint64_t retained = 0;
        std::uint64_t empty_membership = 0;
        std::uint64_t filtered_msc = 0;
        std::uint64_t membership_tokens = 0;

        std::uint64_t min_prevalence = N;
        std::uint64_t max_prevalence = 0;
        std::uint64_t min_minor = N;
        std::uint64_t max_minor = 0;

        const auto t0 =
            std::chrono::steady_clock::now();

        while (true) {
            ssize_t nread = getline(
                &line_ptr,
                &line_capacity,
                pipe
            );

            if (nread < 0)
                break;

            raw_records++;

            while (
                nread > 0 &&
                (
                    line_ptr[nread - 1] == '\n' ||
                    line_ptr[nread - 1] == '\r'
                )
            ) {
                --nread;
            }

            std::string_view line(
                line_ptr,
                static_cast<std::size_t>(nread)
            );

            const std::size_t delim =
                line.find(" |");

            if (delim == std::string_view::npos) {
                throw std::runtime_error(
                    "Missing delimiter at raw record "
                    + std::to_string(raw_records)
                );
            }

            const std::string_view sequence =
                line.substr(0, delim);

            if (sequence.empty()) {
                throw std::runtime_error(
                    "Empty sequence at raw record "
                    + std::to_string(raw_records)
                );
            }

            std::fill(
                packed.begin(),
                packed.end(),
                static_cast<unsigned char>(0)
            );

            std::size_t pos = delim + 2;

            while (
                pos < line.size() &&
                line[pos] == ' '
            ) {
                ++pos;
            }

            std::uint64_t prevalence = 0;

            while (pos < line.size()) {
                while (
                    pos < line.size() &&
                    line[pos] == ' '
                ) {
                    ++pos;
                }

                if (pos >= line.size())
                    break;

                const std::size_t start = pos;

                while (
                    pos < line.size() &&
                    line[pos] != ' '
                ) {
                    ++pos;
                }

                const std::size_t end = pos;

                if (end <= start + 2) {
                    throw std::runtime_error(
                        "Malformed membership token at raw record "
                        + std::to_string(raw_records)
                    );
                }

                const std::string_view token =
                    line.substr(
                        start,
                        end - start
                    );

                // Expected suffix ":1"
                if (
                    token.size() < 3 ||
                    token[token.size() - 2] != ':' ||
                    token[token.size() - 1] != '1'
                ) {
                    throw std::runtime_error(
                        "Unexpected membership state at raw record "
                        + std::to_string(raw_records)
                    );
                }

                const std::string_view sample =
                    token.substr(
                        0,
                        token.size() - 2
                    );

                auto it =
                    sample_to_idx.find(sample);

                if (it == sample_to_idx.end()) {
                    throw std::runtime_error(
                        "Unknown sample at raw record "
                        + std::to_string(raw_records)
                        + ": "
                        + std::string(sample)
                    );
                }

                const std::size_t idx =
                    it->second;

                const std::size_t byte_idx =
                    idx >> 3;

                const unsigned char mask =
                    static_cast<unsigned char>(
                        1u << (idx & 7)
                    );

                if (packed[byte_idx] & mask) {
                    throw std::runtime_error(
                        "Duplicate sample membership at raw record "
                        + std::to_string(raw_records)
                    );
                }

                packed[byte_idx] |= mask;

                prevalence++;
                membership_tokens++;
            }

            if (prevalence == 0) {
                empty_membership++;
                continue;
            }

            if (prevalence > N) {
                throw std::runtime_error(
                    "Prevalence exceeds sample count."
                );
            }

            const std::uint64_t minor =
                std::min<std::uint64_t>(
                    prevalence,
                    N - prevalence
                );

            if (minor < msc_threshold) {
                filtered_msc++;
                continue;
            }

            const std::uint64_t byte_offset =
                retained * bytes_per_feature;

            bout.write(
                reinterpret_cast<const char*>(
                    packed.data()
                ),
                static_cast<std::streamsize>(
                    packed.size()
                )
            );

            if (!bout)
                throw std::runtime_error(
                    "Binary output write failed."
                );

            tout
                << retained << '\t'
                << raw_records << '\t'
                << sequence << '\t'
                << prevalence << '\t'
                << minor << '\t'
                << byte_offset << '\n';

            if (!tout)
                throw std::runtime_error(
                    "Index output write failed."
                );

            min_prevalence =
                std::min(
                    min_prevalence,
                    prevalence
                );

            max_prevalence =
                std::max(
                    max_prevalence,
                    prevalence
                );

            min_minor =
                std::min(
                    min_minor,
                    minor
                );

            max_minor =
                std::max(
                    max_minor,
                    minor
                );

            retained++;

            if (
                retained % 25000 == 0
            ) {
                const auto now =
                    std::chrono::steady_clock::now();

                const double sec =
                    std::chrono::duration<double>(
                        now - t0
                    ).count();

                std::cerr
                    << "[progress] retained="
                    << retained
                    << " raw="
                    << raw_records
                    << " elapsed="
                    << std::fixed
                    << std::setprecision(1)
                    << sec
                    << "s\n";
            }

            if (
                max_features > 0 &&
                retained >= max_features
            ) {
                break;
            }
        }

        if (line_ptr)
            std::free(line_ptr);

        const int gzip_rc =
            pclose(pipe);

        // When max_features is reached early, gzip may receive
        // SIGPIPE because we intentionally stop reading.
        if (
            max_features == 0 &&
            gzip_rc != 0
        ) {
            throw std::runtime_error(
                "gzip reader returned non-zero on full scan."
            );
        }

        bout.flush();
        tout.flush();
        bout.close();
        tout.close();

        if (
            max_features > 0 &&
            retained != max_features
        ) {
            throw std::runtime_error(
                "Benchmark ended before requested retained-feature count."
            );
        }

        const std::uint64_t expected_bin_bytes =
            retained * bytes_per_feature;

        const std::uint64_t actual_bin_bytes =
            fs::file_size(bin_partial);

        if (
            expected_bin_bytes !=
            actual_bin_bytes
        ) {
            throw std::runtime_error(
                "Binary size validation failed."
            );
        }

        fs::rename(
            bin_partial,
            out_bin
        );

        fs::rename(
            tsv_partial,
            out_tsv
        );

        const auto t1 =
            std::chrono::steady_clock::now();

        const double elapsed =
            std::chrono::duration<double>(
                t1 - t0
            ).count();

        std::cout
            << "\n"
            << "============================================================\n"
            << "FILE06 C++ BIT-PACK BENCHMARK\n"
            << "============================================================\n"
            << "Samples                     : "
            << N << "\n"
            << "Bytes per feature           : "
            << bytes_per_feature << "\n"
            << "Raw records scanned         : "
            << raw_records << "\n"
            << "Retained features           : "
            << retained << "\n"
            << "MSC threshold               : "
            << msc_threshold << "\n"
            << "Filtered by MSC             : "
            << filtered_msc << "\n"
            << "Empty-membership records    : "
            << empty_membership << "\n"
            << "Membership tokens parsed    : "
            << membership_tokens << "\n"
            << "Min retained prevalence     : "
            << min_prevalence << "\n"
            << "Max retained prevalence     : "
            << max_prevalence << "\n"
            << "Min retained minor count    : "
            << min_minor << "\n"
            << "Max retained minor count    : "
            << max_minor << "\n"
            << "Binary bytes                : "
            << actual_bin_bytes << "\n"
            << "Elapsed seconds             : "
            << std::fixed
            << std::setprecision(2)
            << elapsed << "\n"
            << "Features / second           : "
            << std::fixed
            << std::setprecision(1)
            << (
                elapsed > 0
                ? retained / elapsed
                : 0.0
            )
            << "\n"
            << "Binary size validation      : PASS\n"
            << "C++ BIT-PACK BENCHMARK      : PASS\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\nFATAL: "
            << e.what()
            << "\n";
        return 1;
    }
}
