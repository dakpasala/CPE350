#!/usr/bin/env python3
"""
extract_first_10M_chars.py
Extracts the first 10 million UTF-8 characters from a huge XML file
without reading the entire file into memory.
"""

INPUT_XML = "../xmls/output11-18-228am.xml"             # your big XML
OUTPUT_XML = "first_10M_chars.xml"    # output
MAX_CHARS = 10_000_000                # 10 million characters

CHUNK_BYTES = 4096                    # read 4KB at a time


def main():
    chars_written = 0
    leftover = b""     # incomplete UTF-8 sequences

    try:
        with open(INPUT_XML, "rb") as fin, open(OUTPUT_XML, "w", encoding="utf-8") as fout:

            while chars_written < MAX_CHARS:
                raw = fin.read(CHUNK_BYTES)
                if not raw:
                    break  # reached end of file

                # safe UTF-8 handling
                raw = leftover + raw

                # try to decode full UTF-8 text
                try:
                    text = raw.decode("utf-8")
                    leftover = b""
                except UnicodeDecodeError as e:
                    # write only valid UTF-8 before error boundary
                    cut = e.start
                    text = raw[:cut].decode("utf-8")
                    leftover = raw[cut:]  # keep fragment for next chunk

                # write only remaining characters needed
                remaining = MAX_CHARS - chars_written
                fout.write(text[:remaining])
                chars_written += len(text[:remaining])

        print(f"Done — wrote {chars_written:,} characters to {OUTPUT_XML}")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
