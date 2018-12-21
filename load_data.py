# class Dataload:
#     def __init__(self):
#         self.datafile = "data/data.txt"
#         self.dataset = self.load_data()
#
#     def load_data(self):
#         dataset = []
#         for line in open(self.datafile):
#             line = line.strip().split(",")
#             dataset.append()

if __name__ == '__main__':
    datafile = "data/data.txt"
    dataset = []
    for line in open(datafile):
        line = line.strip().split(',')
        print(line)