project_path = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/'

src = project_path + 'config.yaml'
tgt = project_path + 'config2.yaml'
with open(src, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
print(src)

converted_lines = [x.replace(
    '/mnt/c/Users/Winst/Documents/',
    '/Users/winston/Documents/XJ/'
).replace(
    '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/',
    './'
) for x in all_lines]

with open(tgt, 'w', encoding='utf-8') as f:
    f.writelines(converted_lines)

print(tgt)