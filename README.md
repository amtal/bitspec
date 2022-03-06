Bit pattern mini-language for instruction encodings. Bytes in, IR out!

[![PyPI Version][pypi-image]][pypi-url]
[![Doctests Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

See [docs](https://amtal.github.io/bitspec) for a step-by-step tutorial and API
reference. Here's a complete example:
```python
>>> import bitspec
>>> @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 00 000 r:3', op='RLC')
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 00 001 r:3', op='RRC')
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 00 010 r:3', op='RL ')
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 00 011 r:3', op='RR ')
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 00 100 r:3', op='SLA')
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 00 101 r:3', op='SRA')
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 00 110 r:3', op='SL1') # "SLL"
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 00 111 r:3', op='SRL')
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 10 b:3 r:3', op='RES')
... @bitspec.bitspec('11 xy:1 11101 0xCB nn:s8 11 b:3 r:3', op='SET')
... class Z80UndocBitOps:    # NORTHERN BYTES Volume 3 #10 (October 1982)
...     def __str__(self):   # mirrored at http://z80.info/z80undoc.htm
...         dst = ['B,', 'C,', 'D,', 'E,', 'H,', 'L,', '', 'A,'][self.r]
...         bit = '' if self.b == None else f'{self.b},'
...         src = ['IX', 'IY'][self.xy]
...         return f'{self.op} {dst}{bit}({src}{self.nn:+})'
...     def __repr__(self): return f'<{self!s}>'
...     def __init__(self, xy, nn, r, op, b=None):
...         self.xy, self.nn, self.r, self.op, self.b = xy, nn, r, op, b

>>> code = bytes.fromhex('fdCB7f17 ddCBfe88 fdCB0125')
>>> Z80UndocBitOps.from_bytes(code)
<RL  A,(IY+127)>

>>> {hex(op.addr):op for op in Z80UndocBitOps.iter_bytes(code, addr=0x50)}
{'0x50': <RL  A,(IY+127)>, '0x54': <RES B,1,(IX-2)>, '0x58': <SLA L,(IY+1)>}
```

Install from [Pypi](https://pypi.org/project/bitspec) or just copy `bitspec.py`
into your project. [Bugs](https://github.com/amtal/bitspec/issues), questions,
or [other feedback](https://github.com/amtal/bitspec/discussions) are welcome!

<!-- Badges -->

[pypi-image]: https://img.shields.io/pypi/v/bitspec
[pypi-url]: https://pypi.org/project/bitspec/
[build-image]: https://github.com/amtal/bitspec/actions/workflows/doctests.yml/badge.svg
[build-url]: https://github.com/amtal/bitspec/actions/workflows/doctests.yml
[coverage-image]: https://codecov.io/gh/amtal/bitspec/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/gh/amtal/bitspec
