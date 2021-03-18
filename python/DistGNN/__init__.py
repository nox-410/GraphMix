def __handle_import():
    import sys
    import os
    cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(cur_path, '../../build/lib/')
    sys.path.append(lib_path)
__handle_import()
