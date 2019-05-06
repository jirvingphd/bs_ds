class LabelLibrary():
    """A Multi-column version of sklearn LabelEncoder, which fits a LabelEncoder
   to each column of a df and stores it in the index dictionary where
   .index[keyword=colname] returns the fit encoder object for that column.

   Example:
   lib =LabelLibrary()

   # Be default, lib will fit all columns.
   lib.fit(df)
   # Can also specify columns
   lib.fit(df,columns=['A','B'])

   # Can then transform
   df_coded = lib.transform(df,['A','B'])
   # Can also use fit_transform
   df_coded = lib.fit_transform(df,columns=['A','B'])

   # lib.index contains each col's encoder by col name:
   col_a_classes = lib.index('A').classes_

   """

    def __init__(self):#,df,features):
        self.index = {}
        from sklearn.preprocessing import LabelEncoder as encoder
        self.encoder=encoder
        # self. = df
        # self.features = features



    def fit(self,df,columns=None):


        if columns==None:
            columns = df.columns
#             if any(df.isna()) == True:
#                 num_null = sum(df.isna().sum())
#                 print(f'Replacing {num_null}# of null values with "NaN".')
#                 df.fillna('NaN',inplace=True)


        for col in columns:

            if any(df[col].isna()):
                num_null = df[col].isna().sum()
                print(f'For {col}: Replacing {num_null} null values with "NaN".')
                df[col].fillna('NaN',inplace=True)

            # make the encoder
            col_encoder = self.encoder()

            #fit with label encoder
            self.index[col] = col_encoder.fit(df[col])


    def transform(self,df, columns=None):
        df_coded = pd.DataFrame()

        if columns==None:
            df_columns=df.columns
            columns = df_columns
        else:
            df_columns = df.columns


        for dfcol in df_columns:
            if dfcol in columns:
                fit_enc = self.index[dfcol]
                df_coded[dfcol] = fit_enc.transform(df[dfcol])
            else:
                df_coded[dfcol] = df[dfcol]
        return df_coded

    def fit_transform(self,df,columns=None):
        self.fit(df,columns)
        df_coded = self.transform(df,columns)
        return df_coded

    def inverse_transform(self,df,columns = None):

        df_reverted = pd.DataFrame()

        if columns==None:
            columns=df.columns

        for col in columns:
            fit_enc = self.index[col]
            df_reverted[col] = fit_enc.inverse_transform(df[col])
        return df_reverted


