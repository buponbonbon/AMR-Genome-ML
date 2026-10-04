#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <sys/types.h>

namespace fs = std::filesystem;

struct State {
    std::uint64_t input_bytes = 0;
    std::uint64_t sample_count = 0;
    std::uint64_t bytes_per_feature = 0;
    std::uint64_t msc = 0;

    std::uint64_t raw_records = 0;
    std::uint64_t retained = 0;
    std::uint64_t empty_membership = 0;
    std::uint64_t filtered_msc = 0;
    std::uint64_t membership_tokens = 0;

    std::uint64_t min_prevalence = 0;
    std::uint64_t max_prevalence = 0;
    std::uint64_t min_minor = 0;
    std::uint64_t max_minor = 0;

    std::uint64_t bin_bytes = 0;
    std::uint64_t tsv_bytes = 0;
};

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
        throw std::runtime_error(
            "Cannot open sample-order file."
        );

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

static void save_state(
    const fs::path& path,
    const State& s
) {
    fs::path tmp = path.string() + ".tmp";

    {
        std::ofstream out(
            tmp,
            std::ios::out | std::ios::trunc
        );

        if (!out)
            throw std::runtime_error(
                "Cannot write checkpoint temp file."
            );

        out << "version=1\n";
        out << "input_bytes=" << s.input_bytes << '\n';
        out << "sample_count=" << s.sample_count << '\n';
        out << "bytes_per_feature=" << s.bytes_per_feature << '\n';
        out << "msc=" << s.msc << '\n';

        out << "raw_records=" << s.raw_records << '\n';
        out << "retained=" << s.retained << '\n';
        out << "empty_membership=" << s.empty_membership << '\n';
        out << "filtered_msc=" << s.filtered_msc << '\n';
        out << "membership_tokens=" << s.membership_tokens << '\n';

        out << "min_prevalence=" << s.min_prevalence << '\n';
        out << "max_prevalence=" << s.max_prevalence << '\n';
        out << "min_minor=" << s.min_minor << '\n';
        out << "max_minor=" << s.max_minor << '\n';

        out << "bin_bytes=" << s.bin_bytes << '\n';
        out << "tsv_bytes=" << s.tsv_bytes << '\n';

        out.flush();

        if (!out)
            throw std::runtime_error(
                "Checkpoint write failed."
            );
    }

    if (
        std::rename(
            tmp.c_str(),
            path.c_str()
        ) != 0
    ) {
        throw std::runtime_error(
            "Atomic checkpoint rename failed."
        );
    }
}

static State load_state(const fs::path& path) {
    std::ifstream in(path);

    if (!in)
        throw std::runtime_error(
            "Cannot open checkpoint file."
        );

    std::unordered_map<
        std::string,
        std::uint64_t
    > values;

    std::string line;

    while (std::getline(in, line)) {
        const auto pos = line.find('=');

        if (pos == std::string::npos)
            continue;

        const std::string key =
            line.substr(0, pos);

        const std::string value =
            line.substr(pos + 1);

        values[key] =
            std::stoull(value);
    }

    if (
        !values.count("version") ||
        values.at("version") != 1
    ) {
        throw std::runtime_error(
            "Unsupported checkpoint version."
        );
    }

    State s;

    s.input_bytes =
        values.at("input_bytes");

    s.sample_count =
        values.at("sample_count");

    s.bytes_per_feature =
        values.at("bytes_per_feature");

    s.msc =
        values.at("msc");

    s.raw_records =
        values.at("raw_records");

    s.retained =
        values.at("retained");

    s.empty_membership =
        values.at("empty_membership");

    s.filtered_msc =
        values.at("filtered_msc");

    s.membership_tokens =
        values.at("membership_tokens");

    s.min_prevalence =
        values.at("min_prevalence");

    s.max_prevalence =
        values.at("max_prevalence");

    s.min_minor =
        values.at("min_minor");

    s.max_minor =
        values.at("max_minor");

    s.bin_bytes =
        values.at("bin_bytes");

    s.tsv_bytes =
        values.at("tsv_bytes");

    return s;
}

static void checkpoint(
    State& state,
    std::ofstream& bout,
    std::ofstream& tout,
    const fs::path& bin_partial,
    const fs::path& tsv_partial,
    const fs::path& state_path
) {
    bout.flush();
    tout.flush();

    if (!bout || !tout)
        throw std::runtime_error(
            "Output flush failed before checkpoint."
        );

    state.bin_bytes =
        fs::file_size(bin_partial);

    state.tsv_bytes =
        fs::file_size(tsv_partial);

    const std::uint64_t expected_bin =
        state.retained
        * state.bytes_per_feature;

    if (state.bin_bytes != expected_bin) {
        throw std::runtime_error(
            "Checkpoint binary-size validation failed."
        );
    }

    save_state(
        state_path,
        state
    );

    std::cerr
        << "[checkpoint] retained="
        << state.retained
        << " raw="
        << state.raw_records
        << " bin="
        << state.bin_bytes
        << " bytes\n";
}

int main(int argc, char** argv) {
    try {
        if (argc != 9) {
            std::cerr
                << "Usage:\n"
                << argv[0]
                << " INPUT.pyseer.gz SAMPLE_ORDER.txt"
                << " OUT.bin OUT.tsv"
                << " MAX_FEATURES MSC"
                << " CHECKPOINT_EVERY STOP_AFTER\n\n"
                << "MAX_FEATURES=0 means full scan.\n"
                << "STOP_AFTER=0 means no deliberate stop.\n";

            return 2;
        }

        const fs::path input_gz   = argv[1];
        const fs::path sample_txt = argv[2];
        const fs::path out_bin    = argv[3];
        const fs::path out_tsv    = argv[4];

        const std::uint64_t max_features =
            std::stoull(argv[5]);

        const std::uint64_t msc =
            std::stoull(argv[6]);

        const std::uint64_t checkpoint_every =
            std::stoull(argv[7]);

        const std::uint64_t stop_after =
            std::stoull(argv[8]);

        if (checkpoint_every == 0)
            throw std::runtime_error(
                "CHECKPOINT_EVERY must be > 0."
            );

        if (!fs::exists(input_gz))
            throw std::runtime_error(
                "Input gzip does not exist."
            );

        if (!fs::exists(sample_txt))
            throw std::runtime_error(
                "Sample-order file does not exist."
            );

        if (
            fs::exists(out_bin) ||
            fs::exists(out_tsv)
        ) {
            throw std::runtime_error(
                "Final output already exists. "
                "Refusing to overwrite."
            );
        }

        auto samples =
            load_samples(sample_txt);

        const std::size_t N =
            samples.size();

        if (N != 4227) {
            throw std::runtime_error(
                "Expected 4227 samples, observed "
                + std::to_string(N)
            );
        }

        const std::size_t bytes_per_feature =
            (N + 7) / 8;

        if (bytes_per_feature != 529)
            throw std::runtime_error(
                "Unexpected bytes-per-feature."
            );

        std::unordered_map<
            std::string_view,
            std::uint16_t
        > sample_to_idx;

        sample_to_idx.reserve(
            N * 2
        );

        for (
            std::size_t i = 0;
            i < N;
            ++i
        ) {
            auto inserted =
                sample_to_idx.emplace(
                    std::string_view(samples[i]),
                    static_cast<std::uint16_t>(i)
                );

            if (!inserted.second) {
                throw std::runtime_error(
                    "Duplicate sample: "
                    + samples[i]
                );
            }
        }

        fs::create_directories(
            out_bin.parent_path()
        );

        fs::create_directories(
            out_tsv.parent_path()
        );

        const fs::path bin_partial =
            out_bin.string() + ".partial";

        const fs::path tsv_partial =
            out_tsv.string() + ".partial";

        const fs::path state_path =
            out_bin.string() + ".checkpoint";

        State state;

        bool resumed = false;

        if (fs::exists(state_path)) {
            if (
                !fs::exists(bin_partial) ||
                !fs::exists(tsv_partial)
            ) {
                throw std::runtime_error(
                    "Checkpoint exists but partial output is missing."
                );
            }

            state =
                load_state(state_path);

            if (
                state.input_bytes
                    != fs::file_size(input_gz) ||
                state.sample_count != N ||
                state.bytes_per_feature
                    != bytes_per_feature ||
                state.msc != msc
            ) {
                throw std::runtime_error(
                    "Checkpoint parameters do not match current run."
                );
            }

            if (
                fs::file_size(bin_partial)
                    < state.bin_bytes ||
                fs::file_size(tsv_partial)
                    < state.tsv_bytes
            ) {
                throw std::runtime_error(
                    "Partial output is smaller than checkpoint."
                );
            }

            // Anything written after the last safe checkpoint
            // is deliberately discarded.
            fs::resize_file(
                bin_partial,
                state.bin_bytes
            );

            fs::resize_file(
                tsv_partial,
                state.tsv_bytes
            );

            resumed = true;

            std::cerr
                << "[resume] retained="
                << state.retained
                << " raw="
                << state.raw_records
                << "\n";
        }
        else {
            if (
                fs::exists(bin_partial) ||
                fs::exists(tsv_partial)
            ) {
                throw std::runtime_error(
                    "Partial output exists without checkpoint. "
                    "Refusing ambiguous recovery."
                );
            }

            state.input_bytes =
                fs::file_size(input_gz);

            state.sample_count = N;
            state.bytes_per_feature =
                bytes_per_feature;

            state.msc = msc;

            state.min_prevalence = N;
            state.min_minor = N;

            {
                std::ofstream bin_init(
                    bin_partial,
                    std::ios::binary
                    | std::ios::trunc
                );

                if (!bin_init)
                    throw std::runtime_error(
                        "Cannot create binary partial."
                    );
            }

            {
                std::ofstream tsv_init(
                    tsv_partial,
                    std::ios::out
                    | std::ios::trunc
                );

                if (!tsv_init)
                    throw std::runtime_error(
                        "Cannot create TSV partial."
                    );

                tsv_init
                    << "Feature Index\tRaw Record\tSequence\t"
                    << "Prevalence\tMinor State Count\t"
                    << "Byte Offset\n";
            }

            state.bin_bytes =
                fs::file_size(bin_partial);

            state.tsv_bytes =
                fs::file_size(tsv_partial);

            save_state(
                state_path,
                state
            );

            std::cerr
                << "[checkpoint] initialized\n";
        }

        std::ofstream bout(
            bin_partial,
            std::ios::binary | std::ios::app
        );

        std::ofstream tout(
            tsv_partial,
            std::ios::out | std::ios::app
        );

        if (!bout || !tout)
            throw std::runtime_error(
                "Cannot open partial outputs."
            );

        std::string command =
            "gzip -dc -- "
            + shell_quote(
                input_gz.string()
            );

        FILE* pipe =
            popen(
                command.c_str(),
                "r"
            );

        if (!pipe)
            throw std::runtime_error(
                "Failed to launch gzip reader."
            );

        char* line_ptr = nullptr;
        std::size_t line_capacity = 0;

        // Resume requires re-reading the compressed stream from
        // the beginning, but already completed records are skipped.
        if (
            resumed &&
            state.raw_records > 0
        ) {
            std::cerr
                << "[resume] fast-forwarding "
                << state.raw_records
                << " raw records...\n";

            for (
                std::uint64_t i = 0;
                i < state.raw_records;
                ++i
            ) {
                const ssize_t nread =
                    getline(
                        &line_ptr,
                        &line_capacity,
                        pipe
                    );

                if (nread < 0) {
                    if (line_ptr)
                        std::free(line_ptr);

                    pclose(pipe);

                    throw std::runtime_error(
                        "Source ended while fast-forwarding "
                        "to checkpoint."
                    );
                }
            }

            std::cerr
                << "[resume] fast-forward complete\n";
        }

        std::vector<unsigned char> packed(
            bytes_per_feature,
            0
        );

        const auto t0 =
            std::chrono::steady_clock::now();

        std::uint64_t next_checkpoint =
            (
                state.retained
                / checkpoint_every
                + 1
            )
            * checkpoint_every;

        bool intentional_stop = false;
        bool target_reached = false;

        while (true) {
            ssize_t nread =
                getline(
                    &line_ptr,
                    &line_capacity,
                    pipe
                );

            if (nread < 0)
                break;

            state.raw_records++;

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

            if (
                delim ==
                std::string_view::npos
            ) {
                throw std::runtime_error(
                    "Missing delimiter at raw record "
                    + std::to_string(
                        state.raw_records
                    )
                );
            }

            const std::string_view sequence =
                line.substr(
                    0,
                    delim
                );

            if (sequence.empty())
                throw std::runtime_error(
                    "Empty sequence at raw record "
                    + std::to_string(
                        state.raw_records
                    )
                );

            std::fill(
                packed.begin(),
                packed.end(),
                static_cast<unsigned char>(0)
            );

            std::size_t pos =
                delim + 2;

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

                const std::size_t start =
                    pos;

                while (
                    pos < line.size() &&
                    line[pos] != ' '
                ) {
                    ++pos;
                }

                const std::string_view token =
                    line.substr(
                        start,
                        pos - start
                    );

                if (
                    token.size() < 3 ||
                    token[token.size() - 2] != ':' ||
                    token[token.size() - 1] != '1'
                ) {
                    throw std::runtime_error(
                        "Unexpected membership token at raw record "
                        + std::to_string(
                            state.raw_records
                        )
                    );
                }

                const std::string_view sample =
                    token.substr(
                        0,
                        token.size() - 2
                    );

                const auto it =
                    sample_to_idx.find(
                        sample
                    );

                if (
                    it ==
                    sample_to_idx.end()
                ) {
                    throw std::runtime_error(
                        "Unknown sample at raw record "
                        + std::to_string(
                            state.raw_records
                        )
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

                if (
                    packed[byte_idx]
                    & mask
                ) {
                    throw std::runtime_error(
                        "Duplicate membership at raw record "
                        + std::to_string(
                            state.raw_records
                        )
                    );
                }

                packed[byte_idx] |= mask;

                prevalence++;
                state.membership_tokens++;
            }

            if (prevalence == 0) {
                state.empty_membership++;
                continue;
            }

            if (prevalence > N)
                throw std::runtime_error(
                    "Prevalence exceeds sample count."
                );

            const std::uint64_t minor =
                std::min<std::uint64_t>(
                    prevalence,
                    N - prevalence
                );

            if (minor < msc) {
                state.filtered_msc++;
                continue;
            }

            const std::uint64_t offset =
                state.retained
                * bytes_per_feature;

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
                    "Binary write failed."
                );

            tout
                << state.retained << '\t'
                << state.raw_records << '\t'
                << sequence << '\t'
                << prevalence << '\t'
                << minor << '\t'
                << offset << '\n';

            if (!tout)
                throw std::runtime_error(
                    "TSV write failed."
                );

            state.min_prevalence =
                std::min(
                    state.min_prevalence,
                    prevalence
                );

            state.max_prevalence =
                std::max(
                    state.max_prevalence,
                    prevalence
                );

            state.min_minor =
                std::min(
                    state.min_minor,
                    minor
                );

            state.max_minor =
                std::max(
                    state.max_minor,
                    minor
                );

            state.retained++;

            if (
                state.retained
                % 25000 == 0
            ) {
                const auto now =
                    std::chrono::steady_clock::now();

                const double sec =
                    std::chrono::duration<double>(
                        now - t0
                    ).count();

                std::cerr
                    << "[progress] retained="
                    << state.retained
                    << " raw="
                    << state.raw_records
                    << " elapsed="
                    << std::fixed
                    << std::setprecision(1)
                    << sec
                    << "s\n";
            }

            if (
                state.retained
                >= next_checkpoint
            ) {
                checkpoint(
                    state,
                    bout,
                    tout,
                    bin_partial,
                    tsv_partial,
                    state_path
                );

                next_checkpoint +=
                    checkpoint_every;
            }

            if (
                stop_after > 0 &&
                state.retained >= stop_after
            ) {
                checkpoint(
                    state,
                    bout,
                    tout,
                    bin_partial,
                    tsv_partial,
                    state_path
                );

                intentional_stop = true;
                break;
            }

            if (
                max_features > 0 &&
                state.retained
                    >= max_features
            ) {
                target_reached = true;
                break;
            }
        }

        if (line_ptr)
            std::free(line_ptr);

        const int gzip_status =
            pclose(pipe);

        if (intentional_stop) {
            bout.flush();
            tout.flush();
            bout.close();
            tout.close();

            std::cout
                << "\nINTENTIONAL CHECKPOINT STOP\n"
                << "Retained : "
                << state.retained
                << "\nRaw      : "
                << state.raw_records
                << "\nCheckpoint preserved for resume.\n";

            return 75;
        }

        if (
            max_features == 0 &&
            gzip_status != 0
        ) {
            throw std::runtime_error(
                "gzip reader returned non-zero "
                "during full scan."
            );
        }

        if (
            max_features > 0 &&
            !target_reached
        ) {
            throw std::runtime_error(
                "Source ended before requested "
                "retained-feature target."
            );
        }

        bout.flush();
        tout.flush();
        bout.close();
        tout.close();

        const std::uint64_t expected_bin =
            state.retained
            * bytes_per_feature;

        const std::uint64_t actual_bin =
            fs::file_size(
                bin_partial
            );

        if (
            actual_bin !=
            expected_bin
        ) {
            throw std::runtime_error(
                "Final binary-size validation failed."
            );
        }

        if (
            std::rename(
                bin_partial.c_str(),
                out_bin.c_str()
            ) != 0
        ) {
            throw std::runtime_error(
                "Could not promote binary partial."
            );
        }

        if (
            std::rename(
                tsv_partial.c_str(),
                out_tsv.c_str()
            ) != 0
        ) {
            throw std::runtime_error(
                "Could not promote TSV partial."
            );
        }

        fs::remove(
            state_path
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
            << "FILE06 RESUMABLE C++ BIT-PACK BUILD\n"
            << "============================================================\n"
            << "Resumed                     : "
            << (resumed ? "YES" : "NO") << "\n"
            << "Samples                     : "
            << N << "\n"
            << "Bytes per feature           : "
            << bytes_per_feature << "\n"
            << "Raw records                 : "
            << state.raw_records << "\n"
            << "Retained features           : "
            << state.retained << "\n"
            << "MSC threshold               : "
            << msc << "\n"
            << "Filtered by MSC             : "
            << state.filtered_msc << "\n"
            << "Empty-membership records    : "
            << state.empty_membership << "\n"
            << "Membership tokens parsed    : "
            << state.membership_tokens << "\n"
            << "Min retained prevalence     : "
            << state.min_prevalence << "\n"
            << "Max retained prevalence     : "
            << state.max_prevalence << "\n"
            << "Min retained minor count    : "
            << state.min_minor << "\n"
            << "Max retained minor count    : "
            << state.max_minor << "\n"
            << "Binary bytes                : "
            << actual_bin << "\n"
            << "Elapsed this invocation (s) : "
            << std::fixed
            << std::setprecision(2)
            << elapsed << "\n"
            << "Binary size validation      : PASS\n"
            << "RESUMABLE BIT-PACK BUILD     : PASS\n";

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
