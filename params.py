def print_usage():
    print("""USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE FILE\n\
    \nDESCRIPTION\
    \n    --train Launch the neural network in training mode. Each chessboard in FILE must
    contain inputs to send to the neural network in FEN notation and the expected output
    separated by space. If specified, the newly trained neural network will be saved in
    SAVEFILE. Otherwise, it will be saved in the original LOADFILE.\
    \n--predict Launch the neural network in prediction mode. Each chessboard in FILE
    must contain inputs to send to the neural network in FEN notation, and optionally an
    expected output.\
    \n--save Save neural network into SAVEFILE. Only works in train mode.\n\n\
    LOADFILE File containing an artificial neural network\n\n\
    FILE File containing chessboards""")

class Params():
    def __init__(self, args):
        self.predict = False
        self.train = False
        self.save = False
        self.help = False
        self.save_file = ""
        self.load_file = "" 
        self.file = ""
        if len(args) < 4:
            print_usage()
            exit(84)
        if args[1] == "--predict" and len(args) == 4:
            if args[2][0] == '-' or args[3][0] == '-':
                print_usage()
                exit(84)
            self.predict = True
            self.load_file = args[2]
            self.file = args[3]
        elif args[1] == "--train" and args[2] != "--save" and len(args) == 4:
            if args[2][0] == '-' or args[3][0] == '-':
                print_usage()
                exit(84)
            self.train = True
            self.load_file = args[2]
            self.file = args[3]
        elif args[1] == "--train" and args[2] == "--save" and len(args) == 6:
            self.save_file = args[3]
            if args[3][0] == '-' or args[4][0] == '-' or args[5][0] == '-':
                print_usage()
                exit(84)
            self.train = True
            self.save = True
            self.load_file = args[4]
            self.file = args[5]
        else:
            print_usage()
            exit(84)