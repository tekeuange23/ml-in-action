from module import hello ## module import
from MainPackage.mainscript import main_report ## package import
from MainPackage.SubPackage.subscript import sub_report ## package import

if __name__ == "__main__":

    hello()
    main_report()
    sub_report()