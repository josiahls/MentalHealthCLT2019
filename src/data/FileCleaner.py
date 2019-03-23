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

    def smart_s2n(st):
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

    def nf1_series(series, flog, CONTINUOUS_COLS, nf1=True, delim=','):
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

                curnumcell = set()

                for e in cellitems:
                    # try to convert to a number, else still the string

                    # then make a new column with the others that didn't work here

                    if not e in items: # change if you want to allow search all for close items
                        # we can pull in from a file comma-separated lines that are the same or on the same scale
                        items.add(e)
                        ind += 1
                        d[e] = ind
                        drev[ind] = e

                    curnumcell.add(d[e])
                numarray[i] = curnumcell
            else:
                # for those that are not 1NF and that are CONTINUOUS_COLS, see if the whole cell can be converted to a float.
                #print(CONTINUOUS_COLS)
                if series.name in CONTINUOUS_COLS:
                    te = FileCleaner.smart_s2n(el)
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

    print(FileCleaner.smart_s2n("$0"))
    assert (FileCleaner.smart_s2n("$0") == 0)

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



