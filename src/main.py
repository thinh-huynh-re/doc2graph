from .data.download import get_data
from .training.funsd import train_funsd
from .utils import project_tree, set_preprocessing
from .training.pau import train_pau

from .argparser import MainArgumentParser


def main():

    args = MainArgumentParser().parse_args()
    print(args)

    if args.init:
        project_tree()
        get_data()
        print("Initialization completed!")

    else:
        set_preprocessing(args)
        if args.src_data == "FUNSD":
            if args.test and args.weights == None:
                raise Exception(
                    "Main exception: Provide a weights file relative path! Or train a model first."
                )
            train_funsd(args)
        elif args.src_data == "PAU":
            train_pau(args)
        elif args.src_data == "CUSTOM":
            # TODO develop custom data preprocessing
            raise Exception(
                'Main exception: "CUSTOM" source data still under development'
            )
        else:
            raise Exception(
                'Main exception: source data invalid. Choose from ["FUNSD", "PAU", "CUSTOM"]'
            )

    return


if __name__ == "__main__":
    main()
