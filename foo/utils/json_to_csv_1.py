from __future__ import print_function
import json


def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                if len(value) == 0:
                    yield pre + [key, '{}']
                else:
                    for d in dict_generator(value, pre + [key]):
                        yield d
            elif isinstance(value, list):
                if len(value) == 0:
                    yield pre + [key, '[]']
                else:
                    if isinstance(value[0], dict):
                        for v in value:
                            for d in dict_generator(v, pre + [key]):
                                yield d
                    else:
                        yield pre + [key, '|'.join(value)]
            elif isinstance(value, tuple):
                if len(value) == 0:
                    yield pre + [key, '()']
                else:
                    for v in value:
                        for d in dict_generator(v, pre + [key]):
                            yield d
            else:
                yield pre + [key, value]
    else:
        yield indict


if __name__ == "__main__":
    fr = open('d:/raw.json')
    sJOSN = fr.read()
    fr.close()
    fw = open('d:/good_data.csv', 'w')
    head = True
    sValue = json.loads(sJOSN)
    if isinstance(sValue, list):
        for line in sValue:
            row = dict_generator(line)
            row_1 = []
            row_2 = []
            for i in row:
                row_2.append(str(i[-1]))
                row_1.append(i[-2])
                # row_1.append('.'.join(i[0:-1]))
            if head is True:
                fw.write(','.join(row_1))
                fw.write('\n')
                head = False
            fw.writelines(','.join(row_2))
            fw.write('\n')
    fw.close()
    print('done')
