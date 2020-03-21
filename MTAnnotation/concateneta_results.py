import argparse
import os
import ntpath


def main():
    """
    concatenate the results file of samples file starts with 'samplesX_'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file_head",
                        type=str,
                        default=None,
                        required=True,
                        help="The batch results csv file head (samplesX_)")
    args = parser.parse_args()
    file_head = ntpath.basename(args.results_file_head)
    directory = os.path.dirname(args.results_file_head)
    res_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(file_head)]
    results = []
    for res_file in res_files:
        with open(res_file, 'r') as fin:
            lines = fin.readlines()
        title = lines[0]
        results += lines[1:]
        results.append('\n')
    with open(args.results_file_head+'results.csv', 'w') as fout:
        fout.write("".join([title] + results))
        print('concatenetaed file results saved to {}'.format(fout.name))

if __name__ == '__main__':
    main()