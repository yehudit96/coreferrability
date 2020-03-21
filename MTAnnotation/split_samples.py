import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file",
                        type=str,
                        default=None,
                        required=True,
                        help="The batch results csv file")
    args = parser.parse_args()
    with open(args.samples_file) as f:
        data = f.readlines()
    head = data[0]
    samples = data[1:]
    samples_split = [samples[i:i+30] for i in range(0, len(samples), 30)]
    for i, split in enumerate(samples_split):
        with open(args.samples_file.replace('.', '_{}.'.format(i+1)), 'w') as fout:
            fout.write("".join([head] + split))


if __name__ == '__main__':
    main()