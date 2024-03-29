# Written by Parker
import pandas as pd
import numpy as np
import os
import sys
import re


class FileCleaner:
    def numerize(series):
        d = {}
        res = [] # pd.Series()
        ind = 0
        for el in series:
            if el in d:
                res.append(d[el])

            else:
                ind += 1
                d[el] = ind
                res.append(ind)

        return pd.Series(res)

    def smart_s2n(st, colname):
        fullmatches={"Yes":1, "No":0, "Associate degree": 4, "Bach.* degree": 5, "Master.* degree": 7, "Professional Degree": 6, "Some college.*": 3, "High school.*":2, "Some high school":0, "Trade.*":1}

        if colname == 'virgin':
            fullmatches = {"Yes": 0, "No": 1} # one-way is different

        for candidate in fullmatches:
            if re.match("^"+candidate+"$", st):
                return fullmatches[candidate]

        stm = st.replace(",", "")
        stm = stm.replace("$", "")
        stm = stm.replace(" ", "")
        try:
            a = float(stm)
            return a
        except:
            pass

        if re.search(" to ", st):
            try:
                a, b = list(map(float, stm.split("to")))
                return (a+b)/2
            except:
                pass

        return st

    def nf1_series(series, flog, CONTINUOUS_COLS, nf1=True, nf1_tight=None, delim=','):
        if nf1_tight == None:
            nf1_tight = nf1

        items = set()
        ind = 0
        d = {}
        drev = {}

        if nf1:
            numarray = [[] for _ in range(len(series))]
        else:
            res = [np.nan for _ in range(len(series))]

        resdf = pd.DataFrame()

        floatseries = None
        isfloatcol = False

        for i,el in enumerate(series):

            if nf1:
                try:
                    cellitems = [e.strip() for e in el.split(delim)]
                except:
                    print(el, series)
                    exit(2)

                if not nf1_tight:
                    curnumcell = set()
                else:
                    curnumcell = []

                for e in cellitems:
                    # try to convert to a number, else still the string

                    # then make a new column with the others that didn't work here

                    if not e in items: # change if you want to allow search all for close items
                        # we can pull in from a file comma-separated lines that are the same or on the same scale
                        items.add(e)
                        ind += 1
                        d[e] = ind
                        drev[ind] = e

                    if not nf1_tight:
                        curnumcell.add(d[e])
                    else:
                        curnumcell.append(d[e])
                numarray[i] = curnumcell
            else:
                # for those that are not 1NF and that are CONTINUOUS_COLS, see if the whole cell can be converted to a float.
                #print(CONTINUOUS_COLS)
                if series.name in CONTINUOUS_COLS:
                    te = FileCleaner.smart_s2n(el, colname=series.name)
                    if isinstance(te, float):
                        if not isfloatcol:
                            floatseries = pd.Series([np.nan for _ in range(len(series))], name=series.name + "_float")
                        isfloatcol = True

                        floatseries[i] = te
                        continue  # this alone fixed this element

                if el in d:
                    res[i] = d[el]

                else:
                    ind += 1
                    d[el] = ind
                    res[i] = ind

        # find a natural order between stuff, and redo the numbers


        if nf1:
            if not nf1_tight:

                assert len(numarray) == len(series)

                for newcoli in range(1, ind+1):
                    newseries = []
                    for j in range(len(series)):
                        if newcoli in numarray[j]:
                            newseries.append(1)
                        else:
                            newseries.append(0)

                    newseries = pd.Series(newseries, name=series.name+"_"+str(newcoli))
                    #newseries.rename(series.name+"_"+str(newcoli)) # using the number for now
                    flog.write((series.name+" "+newseries.name+" "+drev[newcoli]+"\n").encode('utf-8'))


                    resdf = pd.concat([resdf, newseries], axis=1) # horizontal

            else:
                # Represent them as categorical columns
                numnewcols = max(len(cell) for cell in numarray)
                print(numnewcols)
                # see if we can organize this
                # perform more research with better representing columns in 0NF
                for newcoli in range(numnewcols):
                    newseries = []
                    for j in range(len(series)):
                        if newcoli < len(numarray[j]):
                            newseries.append(numarray[j][newcoli])
                        else:
                            newseries.append(-1)
                    resdf = pd.concat([resdf, pd.Series(newseries, name=series.name+"_nf1tight_"+str(newcoli))], axis=1)

        else:
            assert len(res) == len(series)
            resdf = pd.concat([resdf, pd.Series(res,name=series.name+"_categorical")], axis=1)

        if not series.name in CONTINUOUS_COLS: # alt: series.name in CONTINUOUS_COLS or not isfloatcol
            assert not isfloatcol

        if isfloatcol:
            print(series.name + " is float")
            floatseries.fillna(-1, inplace=True) # this should be ok for most
            # DOESNT WORK?
            resdf = pd.concat([resdf, floatseries], axis=1)
        #if series.name == 'income':
        #    print (resdf)

        resdf.fillna(-1, inplace=True)
        return resdf


    # def clean(filename):
    #     df = pd.read_csv(filename)
    #     df.fillna('', inplace=True)
    #     for col in df:
    #
    #         if col in DROP_COLS:
    #             df.drop(col, axis=1)  # drop columns
    #
    #         else:
    #             if not col in LEAVE_COLS:
    #                 try:
    #                     df[col] = pd.to_numeric(df[col])
    #                 except:
    #                     df[col] = FileCleaner.numerize(df[col])
    #
    #     return df

    def shuffle_file(filename_in, filename_out):
        df = pd.read_csv(filename_in)
        df = df.sample(frac=1, random_state=1001).reset_index(drop=True)
        df.to_csv(filename_out, index=False)

    # we could bring these 2 functions together with passing a function as the provider, but leave for now
    def nf1_clean(filename, flog, CONTINUOUS_COLS, delim=','):
        df = pd.read_csv(filename)
        df.fillna('', inplace=True)
        for col in df:
            if not col in LEAVE_COLS:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    if True: # limit to columns that need to be cleaned, rather than any text data with commas
                        if col in NF1_COLS: # NF1 col would never need to be numerized
                            newcolsf = FileCleaner.nf1_series(df[col], flog, CONTINUOUS_COLS, delim=',')
                            df.drop(col, axis=1, inplace=True) # drop columns
                            df = pd.concat([df, newcolsf], axis=1) # horizontal
                        else:
                            try:
                                df[col] = pd.to_numeric(df[col])
                            except:
                                newcolsf = FileCleaner.nf1_series(df[col], flog, CONTINUOUS_COLS, nf1=False)
                                df.drop(col, axis=1, inplace=True)
                                df = pd.concat([df, newcolsf], axis=1)

            if col in DROP_COLS:
                print(col)
                df = df.drop(col, axis=1) # drop columns

        return df

if __name__ == '__main__':
    #print(os.getcwd())
    orig_filepath = "../data/foreveralone.csv"
    filepath = "../data/foreveralone_shuf.csv"
    FileCleaner.shuffle_file(orig_filepath, filepath)

    CONTINUOUS_COLS = ['income']

    flog = open('cleaner.log', 'wb')  # write over
    testlog = open('testcleaner.log', 'wb')  # write over


    # cleanseries = FileCleaner.nf1_series(pd.Series(['frog, cat', 'cat, dog', 'frog'], name="animaltype"), testlog)
    # print(cleanseries)
    cleanseries = FileCleaner.nf1_series(pd.Series(['frog, cat', 'cat', 'frog, cat'],  name="animaltype"), testlog, CONTINUOUS_COLS, nf1=False)
    # print(cleanseries)
    assert (cleanseries .equals( pd.DataFrame(pd.Series([1,2,1], name='animaltype_categorical')) ))

    print(FileCleaner.smart_s2n("$0", "animaltype"))
    assert (FileCleaner.smart_s2n("$0", "animaltype") == 0)
    assert (FileCleaner.smart_s2n("Yes", 'virgin') == 0)
    assert (FileCleaner.smart_s2n("Yes", 'anythingelse') == 1)


    DROP_COLS = ["time", 'job_title']  # need to put in LEAVE_COLS also, not sure why
    LEAVE_COLS = [] + DROP_COLS # leave them as non-numeric to be processed later or dropped here
    NF1_COLS = ['what_help_from_others', 'improve_yourself_how']
    # how do these sort of globals work but CONTINUOUS_COLS doesn't?

    base, name = os.path.split(filepath)
    name_dot_ind = name.rindex('.')
    name_prefix = name[:name_dot_ind]
    name_suffix = name[name_dot_ind:]
    new_indicator = "_cleaned"

    new_filepath = base + "/" + name_prefix + new_indicator + name_suffix

    df = FileCleaner.nf1_clean(filepath, flog, CONTINUOUS_COLS)
    if os.path.isfile(new_filepath):
        sys.stderr.write("Overwriting "+str(new_filepath)+"\n")
    df.to_csv(new_filepath, index=False)

    flog.close()



