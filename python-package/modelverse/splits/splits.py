class Splits(dict):
    def __init__(self, *args, **kw):
        """
        splits = {
                    <split name> : <folds>,
                    <split name> : <folds>,
                    ...
                }
        """
        super().__init__(*args, **kw)

    def __str__(self):
        ret = ""
        for split_name in self.keys():
            ret += f"Split: {split_name}\n\n"
            ret += str(self[split_name])
            ret += '\n'
            ret += '-' * 40 + '\n'
        return ret
