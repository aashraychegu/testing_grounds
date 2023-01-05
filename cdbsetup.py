from watpy.coredb.coredb import *
import pathlib

pathlib.Path("./CoRe_DB").mkdir(exist_ok=True)
cdb = CoRe_db("./CoRe_DB")
cdb.sync(verbose=True, lfs=False)
