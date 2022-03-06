# MIT License
#
# Copyright (c) 2020 amtal
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Bit pattern mini-language for instruction encodings. Bytes in, IR out!

Machine code and interpreter bytecode usually have densely-packed semantic
structure that speeds up tool development when exposed. Specifying encodings
declaratively allows the intermediate representation (IR) to be laid out based
on that structure, not bit-level quirks.

# Example Disassembler

Here's a peculiar group of 4-byte long Zilog Z80 instructions. The architecture
is a byte-prefixed extension of the Intel 8080 crammed into 4 unused opcodes.
Some encoding behaviors were left undefined - possibly to leave room for
further extension.

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

The class decorators add `Bitspec.from_bytes` and `Bitspec.iter_bytes`
classmethods that build pattern-matched objects based on the bitfields
specified. Endianness and signed fields are supported.

>>> code = bytes.fromhex('fdCB7f17 ddCBfe88 fdCB0125')
>>> Z80UndocBitOps.from_bytes(code)
<RL  A,(IY+127)>
>>> {hex(op.addr):op for op in Z80UndocBitOps.iter_bytes(code, addr=0x50)}
{'0x50': <RL  A,(IY+127)>, '0x54': <RES B,1,(IX-2)>, '0x58': <SLA L,(IY+1)>}

The objects get a `__len__` implementation based on which decorator matched.
There's also a `Bitspec.to_bytes` method in case you ever need to patch or
re-assemble code. It's a niche use case, but with a declarative spec it comes
for free!  

>>> Z80UndocBitOps.from_bytes(b'\\xdd\\xcb\\xfe\\x88\\x00\\x00')
<RES B,1,(IX-2)>
>>> i = _; assert len(i) == 4
>>> i.nn = 2; {str(i): i.to_bytes()}
{'RES B,1,(IX+2)': b'\\xdd\\xcb\\x02\\x88'}

# Identifying Structure in an ISA

The data structure above is terse but inconvenient to lift or interpret. We're
already pattern matching once - why switch over the values again in order to
separate simple shifter operations from bit flips? Someone familiar with the Z80's
history would trivially carve it up into octet prefixes and hextet operands.

This library is for incremental tool development on unfamiliar or poorly
documented targets.

## Specializing Instructions

To decode multiple kinds of instructions, group them in a class hierarchy. All
the subclass specifications are merged together and matched at once.  (The
`dataclass` decorator is identical to `bitspec`, but also adds default PEP 557
dataclass methods.)

>>> @bitspec.dataclass
... class HPPA: pass  # Hewlett Packard Precision Architecture (PA-RISC)
>>> @bitspec.dataclass('000010 ----- ----- 0000 001001 0 00000')
... class NOP(HPPA):  #                                  ^^^^^
...     pass          # R0 is always zero. Writing to it is a NOP.
>>> @bitspec.dataclass('000010 00000 r1:5  0000 001001 0 rt:5')
... @bitspec.dataclass('000010 r1:5  00000 0000 001001 0 rt:5')  # hack!
... class COPY(NOP):  #       ^^^^^ ^^^^^
...     r1: int       # If one operand is R0, boolean OR is actually a COPY.
...     rt: int       # To simplify lifting, this IR pretends r2 is r1.
>>> @bitspec.dataclass('000010 r2:5  r1:5  cf:4 001001 0 rt:5')
... class OR(COPY):   # Finally here's the full instruction encoding.
...     cf: int       # You *could* lift OR, COPY, and NOP from it by checking
...     r2: int       # for 0-fields, but those cases are already handled.
>>> import dataclasses; [dataclasses.is_dataclass(c) for c in  OR.__mro__]
[True, True, True, True, False]

Matches are prioritized based on the maximum number of constant bits, with
shortest class distance from the root class as a tie-breaker. This is handy for
architectures where specialized instructions are actually subsets of more
generic ones, or an architecture extension was allocated within a no-op
encoding of an unrelated operation.

>>> HPPA.from_bytes(b'\\x08\\x1f\\x02\\x5f')
COPY(r1=31, rt=31)
>>> HPPA.from_bytes(OR(cf=0, r2=0, r1=31, rt=31).to_bytes())
COPY(r1=31, rt=31)
>>> HPPA.from_bytes(COPY(r1=31, rt=0).to_bytes())
NOP()

## Factoring Operands

While keyword argument values usually get assigned as-is, bitspec-decorated
classes get instantiated with `from_bytes` on the same input as the top-level
object. This allows commonly-used addressing modes and operand types to be
factored into their own classes for easy lifting.

>>> class Operand:  # Intel MCS-51 addressing modes
...     def lift_load(self, il): raise NotImplementedError
...     def lift_store(self, il): raise NotImplementedError
>>> @bitspec.dataclass('.... ....')
... class Acc(Operand): pass                # A
>>> @bitspec.dataclass('.... .... addr:8')
... class Dir(Operand): addr: int           # [addr]
>>> @bitspec.dataclass('.... . reg:3')
... class Reg(Operand): reg: int            # R0..R7
>>> @bitspec.dataclass('.... ... i:1')
... class RegInd(Operand): i: int           # @R0, @R1
>>> @bitspec.dataclass('.:8 val:8')
... class Imm(Operand): val: int            # #val
>>> @bitspec.dataclass('.:16 val:8')
... class Imm_(Imm): pass  # re-use lift semantics on a new encoding!
>>> @bitspec.dataclass('0100 0100', name='ORL', dst=Acc, src=Imm)
... @bitspec.dataclass('0100 0101', name='ORL', dst=Acc, src=Dir)
... @bitspec.dataclass('0100 011.', name='ORL', dst=Acc, src=RegInd)
... @bitspec.dataclass('0100 1...', name='ORL', dst=Acc, src=Reg)
... @bitspec.dataclass('0100 0010', name='ORL', dst=Dir, src=Acc)
... @bitspec.dataclass('0100 0011', name='ORL', dst=Dir, src=Imm_)
... @bitspec.dataclass('0101 0100', name='ANL', dst=Acc, src=Imm)
... @bitspec.dataclass('0101 0101', name='ANL', dst=Acc, src=Dir)
... @bitspec.dataclass('0101 011.', name='ANL', dst=Acc, src=RegInd)
... @bitspec.dataclass('0101 1...', name='ANL', dst=Acc, src=Reg)
... @bitspec.dataclass('0101 0010', name='ANL', dst=Dir, src=Acc)
... @bitspec.dataclass('0101 0011', name='ANL', dst=Dir, src=Imm_)
... # [...] repetitive code for XRL, ADD, ADDC, etc.
... class MCS51ALU: 
...     name: str
...     dst: Operand
...     src: Operand

The resulting `__len__` is adjusted based on any arguments `from_bytes` was
called on, so variable-length encodings work as you'd expect.

>>> code = bytes.fromhex('438010 58 52ff')
>>> for i in MCS51ALU.iter_bytes(code, addr=0):
...     print(f'{hex(i.addr)}({len(i)}): {i}')
0x0(3): MCS51ALU(name='ORL', dst=Dir(addr=128), src=Imm_(val=16))
0x3(1): MCS51ALU(name='ANL', dst=Acc(), src=Reg(reg=0))
0x4(2): MCS51ALU(name='ANL', dst=Dir(addr=255), src=Acc())

## Incremental Development

What if we're implementing tools for an unfamiliar architecture, and it's not
obvious how addressing modes are encoded?

Suppose we thought the MCS-51 instruction set was orthogonal and operand
addressing mode encodings were independent. The preceding code could be
code-golfed by having the operand arguments be an orthogonal pattern match.

But wait - MCS-51 isn't orthogonal! One of the MOV encodings that would be a
NOP on an architecture with bits to spare is actually a JMP, and another had a
bit-level ORL variant squeezed in.

>>> @bitspec.dataclass('.... 0100 n:8', dst='A', src='#{n}')     # immediate
... @bitspec.dataclass('.... 0101 n:8', dst='A', src='0x{n:x}') # direct
... @bitspec.dataclass('.... 011 n:1',  dst='A', src='@R{n}' ) # I-ram
... @bitspec.dataclass('.... 1 n:3',    dst='A', src='R{n}')  # register
... @bitspec.dataclass('0111 0010 n:8', dst='C', src='{n}')  # bit addr (hack)
... @bitspec.dataclass('.... 0011 n:8 m:8', dst='0x{n:x}', src='#{m}')
... @bitspec.dataclass('.... 0010 n:8',     dst='0x{n:x}', src='A')
... class Operands:
...     src:str; dst:str; n:int = -1; m:int = -1
...     def __str__(self):
...         return f'{self.dst},{self.src}'.format(**self.__dict__)
...     def __repr__(self): return f'<{len(self)}: {self!s}>'
>>> @bitspec.dataclass('0100 ....', name='ORL', ops=Operands)
... @bitspec.dataclass('0101 ....', name='ANL', ops=Operands)
... @bitspec.dataclass('0110 ....', name='XRL', ops=Operands)
... @bitspec.dataclass('0111 ....', name='MOV', ops=Operands)
... # [...]
... @bitspec.dataclass('0111 0011', name='JMP', ops='@A + DPTR') # (hack)
... class OrthMCS51ALU:
...     name:str; ops:Operands
...     def __repr__(self): return f'<{len(self)}: {self.name} {self.ops!s}>'

And that's fine. Since `Bitspec.from_bytes` only directly pattern matches on
the class hierarchy it is called on, with any operands being pattern matched
only after the first-level match succeeds, some conflicts and overlaps with
bitspec arguments are okay. 

>>> OrthMCS51ALU.from_bytes(b'C'*3)
<3: ORL 0x43,#67>
>>> jmp_indirect = b'\\x73' + b'C' * 10
>>> OrthMCS51ALU.from_bytes(jmp_indirect)  # correct disassembly
<1: JMP @A + DPTR>
>>> Operands.from_bytes(jmp_indirect)  # top-level match takes priority!
<3: 0x43,#67>

A disassembler and lifter can be quickly brought up based on common encoding
structures. Edge cases can be filled in later as they're figured out.

## Debugging and Partial Decoding

Bringing up a new ISA target will often raise questions:

* "Am I looking at code or data? The disassembly doesn't make sense."
* "Did I typo this instruction's encoding or is the reference manual wrong?"
* On bad days, "Does my memory dump contain semi-random bitflips?"

This library can't fix signal integrity in 1' of 0.1" jumper wire spaghetti. It
*can* disassemble just one class of instructions at a time, which helps debug
encoding typos from an interpreter.

>>> NOP.from_bytes(b'\\x08\\x1f\\x02\\x41')
COPY(r1=31, rt=1)

You can even peek under the hood for a quick encoding reference while doing it.

>>> from pprint import pprint as pp
>>> pp(COPY.__bitspec__)
{<Match 000010.....0000000000010010..... /4>: {'r1': <Slice 0:6 u:5 _:21 /4>,
                                               'rt': <Slice 0:27 u:5 /4>},
 <Match 00001000000.....00000010010..... /4>: {'r1': <Slice 0:11 u:5 _:16 /4>,
                                               'rt': <Slice 0:27 u:5 /4>}}
>>> pp(NOP.__bitspec__)
{<Match 000010..........0000001001000000 /4>: {}}

There's also a sometimes-useful (sometimes-annoying) edge case when decoding
short bytestrings on variable length architectures: it's possible to decode an
instruction prefix, but not have enough bytes left for the operand.

>>> OrthMCS51ALU.from_bytes(b'\\x72' + b'C')  # ok
<2: MOV C,67>
>>> OrthMCS51ALU.from_bytes(b'\x72')  # no operand!
<1: MOV None>

Usually that's a sign the ISA specification is missing an instruction or needs
the top-level match padded out to a full instruction length. However, at the
end of a basic block or data section it's a good sign you're disassembling
data.

# Motivation and Similar Tools

Isolating bit-level encoding details to a flexible mini-language allows the
low-level IR to be designed entirely around the target ISA's semantic quirks
rather than its encoding quirks. Building around addressing modes, ALU design,
memory banking schemes, or significant ISA extensions greatly speeds up lifter
bringup, debugging, and maintenance.  

Q: But I wanna go *fast*! Why not just write table-based (dis)assemblers?  <br>
A: You should - in Rust. Python's GIL and ctypes back-and-forth to your
   binary analysis framework of choice will limit scalability anyway.

Q: Isn't this overthinking a problem that doesn't exist?                   <br>
A: In theory the number of ISAs in this world is finite and we can
   brute-force our way to nice tool support for all of them. 
   In practice appearance of interesting targets, weird bytecode machines,
   and binary analysis platforms seems to outpace tool publications.

Q: The DSL style looks familiar.                                           <br>
A: See GDB's opc2c.c, QEMU's decodetree.py, or Amoco's @ispec as other examples.
   
Q: Do all sufficiently complex binary analysis projects really
   contain an ad-hoc implementation of LLVM's MCInst?                      <br>
A: Yes, but only touch LLVM when paid to.
"""
## read docs locally with `python -m pydoc -b bitspec`
## (or ideally with the `pdoc3` package, but keep it out of CI? hmm)
__version__ = '0.4.4'
__author__ = 'amtal'
__license__ = 'MIT'  # https://opensource.org/licenses/MIT
__all__ = ['dataclass', 'bitspec', 'is_bitspec', 'Bitspec']
# TODO top-level from_bytes API rework
#       addr=n argument to iter_bytes not present on from_bytes, weird
#       maybe drop from_bytes, add a fromhex to keep examples next()-free?
# TODO  want a len_iter_bytes length-decoder fastpath?
#        - need a way to signal full-decoding need though
#          since you usually want branch decoding not just length
#        - also very inconvenient to do with current arglen behavior,
#          arguments might expand final matched length so whole match tree is
#          too dynamic to try and extract a closed-form solution
#        - __bitspec_match__ is also a hack currently breaks "frozen" objects,
#          maybe dynamic length isn't worth it even in Python
# TODO examples of NamedTuple / other __slots__-based IRs
# TODO re-examine python version high watermark
# TODO changelog in comments?
from itertools import groupby
# Python >= 3:
import inspect 
from types import FunctionType
import typing  # Python >= 3.5 for NamedTuple subclass
import abc, collections.abc


# Attribute name for storing bitfield spec, canonical indicator for is_bitspec.
_SPEC = '__bitspec__'
# Cache of all subclasses, updated when __hash__ changes or on first use?
_CACHE = '__bitspec_cache__'
# Which pattern an object was matched from. (Used only in re-assembly.)
_MATCH = '__bitspec_match__'


class Bitspec(collections.abc.Sized, metaclass=abc.ABCMeta):
    """ABC[^1] for type-annotating `bitspec`-decorated classes.

    >>> import typing
    >>> @bitspec.dataclass('0x414141')
    ... class Aaa: pass
    >>> def decode(bs: bytes) -> typing.List[bitspec.Bitspec]:
    ...     return list(Aaa.iter_bytes(bs))

    The library follows the same extension pattern as PEP 557 dataclasses. No
    new class is created. As a result, inheritance is untouched!

    [^1]: Not a real PEP 3119 ABC, just here for annotations and docs. :)

    >>> xs = decode(b'A' * 9); xs
    [Aaa(), Aaa(), Aaa()]
    >>> b''.join(x.to_bytes() for x in xs)  # this'll typecheck nicely, though
    b'AAAAAAAAA'
    >>> assert not any(isinstance(x, bitspec.Bitspec) for x in xs)

    *[ISA]: Instruction Set Architecture
    *[IR]: Intermediate Representation
    *[ALU]: Arithmetic Logic Unit
    *[DSL]: Domain Specific Language
    *[GIL]: Global Interpreter Lock
    *[ABC]: Abstract Base Class
    *[PEP 557]: dataclasses module added in Py3.7
    *[PEP 3119]: Introducing Abstract Base Classes
    """
    # Dataclass-style extension doesn't actually re-create the class, so we
    # can't add an extra parent. This is purely a pydoc / mypy hallucination.
    # Could add it as a decoration?
    #
    # For type annotation reasons, it has to be declared here and defined at
    # the bottom. Every release of Python strays further from God's light.
    __slots__ = _SPEC, _MATCH
    from_bytes:classmethod
    iter_bytes:classmethod
    addr:typing.Optional[int]
    """Address of decoded instruction.
    
    Optional, only set if a base `addr` is passed to `Bitspec.iter_bytes`.
    """
    to_bytes:typing.Callable[[typing.Any, int], bytes]
    __len__:typing.Callable[[typing.Any], int]


def bitspec(specification='', **const_fields):
    """Class decorator which adds `Bitspec` methods to an existing class.

    Mini-language grammar, with whitespace separating constants and fields:

        specification := (const_bits | const_bytes | variable)+
           const_bits := ('.' | '0' | '1')+
          const_bytes := '0' 'x' hex_value
             variable := (py_identifier | '.') ':' 's'? int_size

    Bits and bytes are indexed in big-endian order, with the most significant
    bit at the start of the specification and bit 0 at the end. The byte order
    can be flipped for little-endian memory platforms, but bit order remains
    the same.
    
    Any bitspec argument fields are part of the full width; don't-care bits
    will be appended to the end of any too-short declarations.
    
    >>> # byte-endianness:  [      3] [      2] [    1] [0]
    >>> # bit-endianness:   31-----24 23-----16 15----8 7-0
    >>> @bitspec.dataclass('.... .... .... .... a:4 b:4 c:8')
    ... class Op: 
    ...     a: int; b: int; c: int
    >>> @bitspec.dataclass('1000 0100 0010 0001', op=Op)
    ... class BitOrder: 
    ...     op: Op
    >>> BitOrder.from_bytes(b'\\x84\\x21\\x53\\x10')
    BitOrder(op=Op(a=5, b=3, c=16))
    >>> BitOrder.from_bytes(b'\\x21\\x84\\x10\\x53', byteswap=2)
    BitOrder(op=Op(a=5, b=3, c=16))

    Total size should be aligned on byte boundaries, but internal bit fields
    can have arbitrary widths and alignments. 

    Syntax sugar includes:

    - An empty spec (default) can't be matched, but still gets all the extra
      methods. This is often the case for top-level "instruction" classes that
      anchor multiple subclassed instruction types. Either `@bitspec()` or
      `@bitspec` decorator syntax can be used.
    - Long runs of ..... don't-care bits can be listed as a '.' or '-' variable.
    - Wildcard characters '.' and '-' are completely interchangeable. This can
      be handy to differentiate between actual don't-care bits and parts of the
      pattern match that are matched/extracted in an argument, but is purely a
      hint to the reader and isn't validated in any way.
    - Byte fields for instruction sets with known prefixes or big constants.
      They have to be byte-aligned and a multiple of 8 bits wide.

    >>> @bitspec.dataclass('-:7 .:9 a:4 b:4 c:8')
    ... class ShortOp(Op): pass
    >>> @bitspec.dataclass('0x8421', op=ShortOp)
    ... class ShortBitOrder(BitOrder): pass
    >>> ShortBitOrder.from_bytes(b'\\x84\\x21\\x53\\x10')
    ShortBitOrder(op=ShortOp(a=5, b=3, c=16))

    Ambiguities are resolved by:
    
    1. Maximizing the number of constant bits matched.
    2. Prioritizing the shallowest reachable class. (This is why the above
       example isn't ambiguous, even though BitOrder is part of its pattern
       match.)

    Prefixing field size with 's' will read and sign-extend a 2s complement
    signed value.

    >>> @bitspec.dataclass('imm:32 off:s16 src:4 dst:4 op:8')
    ... class EBPF: 
    ...     imm:int; off:int; src:int; dst:int; op:int
    >>> ja_neg4 = bytes.fromhex('05 00 fcff 00000000')
    >>> EBPF.from_bytes(ja_neg4, byteswap=8)
    EBPF(imm=0, off=-4, src=0, dst=0, op=5)

    .. todo:: Multiple fields with the same name will be concatenated.

    Detection of dead code due to ambiguous or over-constrained specifications
    is best-effort and not guaranteed; this is an instruction decoder not a
    general-purpose parser. That said, the load-time checks provide a bit more
    assurance than usual for a statically unityped language like Python.

    Raises:
        SyntaxError: certain bugs (e.g. field name not a valid variable)
        SyntaxWarning: suspected bugs (e.g. ambiguous or overconstrained specs)
        IndexError: top-level byte alignment violated
        NameError: field names don't match constructor arguments
    """
    if callable(specification) and len(const_fields) == 0:  # @bitspec
        return install_methods(specification, None, {}, {})

    match, var_fields = load_time_parse(specification)
    check_duplicate_args(var_fields, const_fields)
    def add_bitspec(cls):
        check_class_args(cls, var_fields, const_fields, specification)
        return install_methods(cls, match, var_fields, const_fields)
    return add_bitspec


def test_bitspec():   # hack to make doctest see tests despite function
    pass              # clashing with "import" glob (breaks line #s, oh well)
test_bitspec.__doc__ = bitspec.__doc__


import dataclasses  # Python >= 3.7 (maybe ImportError gate @dataclass?)
def dataclass(specification='', **const_fields):
    """Same class decorator as @`bitspec` but with a PEP 557 @dataclass inside.

    You'll still need the extra line if using non-default dataclass arguments.
    
    >>> import dataclasses
    >>> @bitspec.bitspec('0xf000')
    ... @dataclasses.dataclass  # or just do @bitspec.dataclass('0xf000')
    ... class Foo: pass
    """
    if callable(specification) and len(const_fields) == 0:  # @dataclass
        return install_methods(install_dataclass(specification), None, {}, {})

    match, var_fields = load_time_parse(specification)
    check_duplicate_args(var_fields, const_fields)
    def add_bitspec_with_dataclass(cls):
        cls = install_dataclass(cls)
        check_class_args(cls, var_fields, const_fields, specification)
        return install_methods(cls, match, var_fields, const_fields)
    return add_bitspec_with_dataclass


def install_dataclass(cls):
    # dataclasses.is_dataclass returns true for *inherited* dataclasses due
    # to using `hasattr`, we're just looking to see if @dataclass has been
    # applied already to avoid re-running it.
    is_already = dataclasses._FIELDS in cls.__dict__
    return cls if is_already else dataclasses.dataclass(cls)
    # this was the only use of the module, could drop dep for py2.7


def is_bitspec(cls_or_obj: typing.Any) -> bool:
    """True if `cls_or_obj` has been directly decorated with a `bitspec`.
    
    >>> @bitspec.bitspec
    ... class Foo: pass       # can be matched, has extra methods
    >>> class Bar(Foo): pass  # can't be matched, no extra methods
    >>> @bitspec.bitspec("0x0badf00d")
    ... class Baz(Bar): pass  # part of the match for Foo
    >>> [bitspec.is_bitspec(x) for x in [Foo, Bar, Baz]]
    [True, False, True]
    >>> [cls.from_bytes(b'\\x0b\\xad\\xf0\\x0d') for cls in Baz.__mro__ 
    ...                                          if bitspec.is_bitspec(cls)]
    [<__main__.Baz object at ...>, <__main__.Baz object at ...>]

    Since `Bitspec.from_bytes` returns instances of the specific class that's
    been decorated, there's no clear meaning for calling it on a non-decorated
    subclass. As such while bitspec methods may resolve on a subclass, calling
    them is currently an error.

    >>> try: Bar.from_bytes(b'\\xde\\xad\\xbe\\xef')
    ... except TypeError: True
    True
    """
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    return _SPEC in cls.__dict__

def iter_bytes(cls, bytes:bytes, byteswap=0, addr=None) -> typing.Iterable[Bitspec]:
    """Generate a sequence of objects pattern-matched from bytes.

    Yields results until a match fails. Un-decoded bytes can be identified
    based on last-decoded instruction address:

    >>> @bitspec.bitspec('0x41414141')
    ... class AAA: pass
    >>> mash, start = b'A'*1024 + b'\\xde\\xad\\xbe\\xef', 0x8000
    >>> for a in AAA.iter_bytes(mash, addr=start):
    ...     pass
    >>> mash[a.addr - start + len(a):]
    b'\\xde\\xad\\xbe\\xef'

    Args:
        byteswap: little-endian word width in bytes, 0 for big-endian
        addr: set an `addr` attribute on generated objects, incrementing it as
            generator advances

    Raises:
        TypeError: no bitspec decorator found on cls
    """
    # perform byte-endianness swap ASAP, if matches aren't word-aligned at
    # least the resulting bugs won't be confusing
    bytes = swap_endianness(bytes, byteswap)

    remaining = len(bytes)
    while remaining:
        if obj := from_bytes(cls, bytes[-remaining:]):
            if addr != None:
                obj.addr = addr
                addr += len(obj)
            remaining -= len(obj)
            yield obj
        else:  # remaining length can be reconstructed from obj.addr
            return


def reachable_bitspec_classes(root):
    # enumerate all possible subclasses that are part of pattern match
    # breadth-first traversal storing shortest path as later tie-breaker
    #
    # if A->B->C and is_bitspec(x) is True,False,True then C isn't reached...
    # document? full traverse?
    cls_tree = {}
    level = {root,}
    next_level = set()
    depth = 0
    while level:
        for cls in level:
            # Subtle: traverse *entire* class tree, but only return bitspec
            # matches. This means A->B->C where only A and C are decorated will
            # match both in A.from_bytes, but ignores B.
            #
            # This can come up when doing complicated things with operands for
            # the sake of code-golfing a lifter into something legible.
            if cls not in cls_tree and is_bitspec(cls):
                cls_tree[cls] = depth
            {next_level.add(sub_cls) for sub_cls in cls.__subclasses__()}
        level, next_level = next_level, set()
        depth += 1
    max_depth = depth
    return cls_tree, max_depth


import functools
@functools.lru_cache(maxsize=512)
def __precompute(cls):
    cls_tree, max_depth = reachable_bitspec_classes(cls)
    possible_matches = []
    for a_cls, depth in cls_tree.items():
        for match in a_cls.__dict__[_SPEC]:
            score = bin(match.mask).count('1')  # popcnt
            result = (score, max_depth - depth, match, a_cls)
            possible_matches.append((match, result))
    opaque = Match.multimatch_precompute(possible_matches)
    return opaque

@functools.lru_cache(maxsize=4096)
def from_bytes(cls, bytes: bytes, byteswap=0) -> typing.Optional[Bitspec]:
    """Constructor classmethod.

    Args:
        byteswap: little-endian word width in bytes, 0 for big-endian

    Returns: 
        `None` if match unsuccessful due to insufficient bytes or wrong prefix.

    Raises:
        TypeError: no bitspec decorator found on cls
    """
    if not is_bitspec(cls):
        msg = f'''{cls} has not been decorated by @bitspec directly
        
Either the intent was to match subclasses ({cls.__subclasses__()})
in which case {cls} needs a trivial decorator, or a decorated class
(one of {cls.__mro__[1:-1]}) 
has been subclassed. Since bitspec from_bytes returns very specific classes on
successful matches, deserializing non-bitspec subclasses do not make sense.
'''
        raise TypeError(msg)

    opaque = __precompute(cls)

    bytes = swap_endianness(bytes, byteswap)
    if not (possible_matches := Match.multimatch_execute(opaque, bytes)):
        return None
    _, _, match, matched_cls = max(possible_matches)

    # build object
    def slice_off_argument(val):
        if isinstance(val, Slice):
            return val.from_bytes(bytes)
        elif isinstance(val, type) and is_bitspec(val):
            # ^ Check that it was an uninstantiated class, not an instance.
            # Otherwise passing an initialized object will silently
            # replace it with a brand-new one, initialized from the bytes.
            return val.from_bytes(bytes)
        else:
            return val
    spec_args = matched_cls.__dict__[_SPEC][match]
    kwargs = {name:slice_off_argument(spec_args[name]) for name in spec_args}
    obj = matched_cls(**kwargs)

    # adjust length for matched args; they're not part of the pattern match,
    # but not propagating length up means user would have to max(len(i),
    # len(i.src), len(i.dst)) or something dumb
    matched_length = max([match.byte_length] + [len(k) for k in kwargs.values()
                                                if is_bitspec(k)])
    if matched_length != match.byte_length:
        match = match.expand_by(matched_length - match.byte_length)
    setattr(obj, _MATCH, match)  # for easy `to_bytes`, not slots-compatible
    return obj

# Matches are prioritized on number of constant bits first, matched class
# distance from root second.
# 
# **Design Rationale:**
#
# Class depth is a clean tie-breaker for slicing out operand fields, which
# usually has no constant-bits, only don't-care bits and field slices. As
# such it's common to have a complex class hierarchy you want to decode
# from bytes, but none of the classes actually have constant bit matches to
# disambiguate them. This behavior guarantees the "nearest" class gets
# decoded, which usually ends up being the one the decode classmethod
# was called on.
#
# It's an ugly tie-breaker for rare cases where you might want to decode
# macroinstructions or special-cased instruction aliases. The PA-RISC
# example in the module docs does this to decode a no-op encoding as
# a specific NOP, although practically there are multiple no-op encodings
# and which one gets used by a given target's assembler isn't guaranteed.
#
# There might be CISC var-length architectures where the max-number of
# constant bits heuristic isn't valid and you need a proper parser. Can't
# think of any off the top of my head, easily solved via a first-pass
# decoder that looks for prefix bytes or whatever and dispatches to three
# or four appropriate instruction types.
#
# **Alternatives Considered:**
#
# * manually prioritizing conflicts via foo.decorator, too much effort
#   for common case
# * passing a floating point weight-adjustment as an extra arg, too much
#   internal details leakage + might collide with field names
#
# tl;dr this is where not treating this as a proper parser bites us,
# but afaict it's still worth it and full LR(8) bitparsing or w/e is
# complete overkill


# Bitspec arguments affect the pattern match by increasing its byte length.
# Bitspec arguments do not affect the matched constant values, failure to match
# argument constants just results in that argument returning None.
#
# This should be fairly easy to debug (AttributeError: `NoneType` object has
# no attribute 'lift`) and should only really happen if the argument
# does some complex operand dispatch... But it's usually easier to
# specify operand types in a top-level instruction pattern, than to
# figure out operand encoding invariants that hold across all
# instruction classes and push them down to an operand argument.
#
# tl;dr Bitspec pattern matches traverse subclasses, but not arguments.
# This is an implementation detail that also simplifies fast matching,
# but should still be documented somewhere?

import array
def swap_endianness(bytes, word_length):
    """Silently truncates non-word-sized inputs."""
    if not word_length:
        return bytes
    elif word_length in [2,4,8]:
        if tail := len(bytes) % word_length:
            bytes, tail = bytes[:-tail], bytes[-tail:]

        arr = array.array({2:'H',4:'I',8:'Q'}[word_length])
        arr.frombytes(bytes)
        arr.byteswap()
        return arr.tobytes()  # Py>=3.2 lol
    else:
        msg = f'{word_length*8}-bit byte endianness not implemented'
        raise NotImplementedError(msg)


def to_bytes(self: Bitspec, byteswap=0) -> bytes:
    """Assemble IR result of `Bitspec.from_bytes`.
    
    Only works if object has fields that exactly match its constructor
    arguments. This is a common Python convention and is true for PEP 557
    dataclasses, but if you don't intend to use this method you can completely
    ignore the convention.
    
    Ambiguities are resolved as follows:

    - Don't-care bit positions will be set to zero.
    - Fixed bits will be set to whatever they were originally decoded from.
    - If the object was constructed manually rather than via from_bytes, 
      fixed bits will be chosen from an arbitrary match decorator. If there are
      multiple decorators, exact one chosen is undefined.

    As an added benefit, this means instantiated bitspec arguments can be
    passed to pattern matches. A common example would be specializing a
    particular encoding. Doing so isn't a meaningful optimization, but it's
    nice to have it "just work"[^1] rather than throw errors. 
    
    [^1]: This is probably a nitpicky implementation detail, but the provided
    example *could* return a length of 1 since `SReg(0)` was never pattern
    matched and might have valid definitions with other lengths? Current
    solution is dead simple and just treats len(arg.to_bytes()) as a lower
    bound on length. FIXME rm docs leave test

    >>> @bitspec.dataclass('.:8 n:4 ....')
    ... class SReg: n:int
    >>> @bitspec.dataclass('0x01 .... imm:s4', r=SReg,    op='load-rel')
    ... @bitspec.dataclass('0x01 0000 imm:4',  r=SReg(0), op='load-abs')
    ... class SIns: r:SReg; op: str; imm:int
    >>> SIns.from_bytes(b'\\x01\\x0f')
    SIns(r=SReg(n=0), op='load-abs', imm=15)
    >>> len(_.r)  # re-calculated off of .to_bytes(), not cached!
    2
    
    Disassembly is expected to be more performance-sensitive than re-assembly, so
    the implementation is likely to be slower than from_bytes.
    """
    try:
        match = getattr(self, _MATCH)
    except AttributeError:
        # fully synthetic object constructed w/o matching
        # FIXME actually walk down matches to find correct match
        match = list(getattr(self.__class__, _SPEC).keys())[0]  # lol guess
    fields = getattr(self.__class__, _SPEC)[match]

    acc = match.const
    for name in fields:
        field = fields[name]
        if isinstance(field, Slice):
            val = getattr(self, name)  # assume field names == __init__ args
            val <<= field.shift
            acc |= val

    big_endian = acc.to_bytes(length=match.byte_length, byteorder='big')
    return swap_endianness(big_endian, byteswap)


def byte_length(self: Bitspec) -> int:
    """Byte length of matched value.
    
    If the object wasn't built by calling `from_bytes` or `iter_bytes`, length
    should still be correct in simple cases. Multiple matches of variable
    lengths *might* result in wrong length being returned.
    """
    if match := getattr(self, _MATCH, None):
        return match.byte_length
    else:
        # Synthetic-constructed instance; realistically, most likely reason to
        # check len() on this is the default truthiness implementation. e.g.
        # testing an operand with a default value of None will call __len__ if
        # it's present.
        # So, actual accurate length probably doesn't matter so just guess :)
        # If multiple lengths are possible lol good luck. 
        # FIXME actually walk down matches to find correct match
        assert type(self) != type
        match = list(getattr(self.__class__, _SPEC).keys())[0] # lol guess
        return match.byte_length


BIT_CONST_CHARS = set('01.-')
def load_time_parse(specification):
    """Parses spec into a match pattern and dict of variables.

    The hand-rolled parser implementation is super-gnarly, but that's okay if
    it never leaks out of this function. This is a key source of user-facing
    error messages so it's worth keeping the exception call stack minimal.

    Gnarly implementation regression-test hall of shame:

    >>> parse = bitspec.load_time_parse
    >>> assert (x := parse('....0001')) == parse('.... 0001'); x
    (<Match ....0001 /1>, {})
    >>> assert (x := parse('0x0008')) != parse('0x08'); x
    (<Match 0000000000001000 /2>, {})
    """
    spec_width = 0  # running total in bits
    fields = []  # tagged-tuple IR := ('c' index mask const bit_width)
                 #                  | ('v' index mask name  is_signed)

    # parse spec, indexing bit-offsets starting from zero @ most significant
    # (will get re-indexed later once full width is known)
    for token in specification.split():
        if set(token).issubset(BIT_CONST_CHARS):
            const_value = int('0b' + token.replace('.','0')
                                          .replace('-','0'), 2)
            const_mask = int('0b' + token.replace('0','1')
                                         .replace('.','0')
                                         .replace('-','0'), 2)
            const_width = len(token)
            fields.append(('c', spec_width, const_mask, const_value, const_width))
            spec_width += const_width
        elif token.startswith('0x'):
            const_value = int(token, 16)
            const_width = len(token[2:]) * 4
            const_mask = (1 << const_width) - 1
            if len(token) == 3:
                suggest_a = token[:2] + '0' + token[2]
                suggest_b = bin(const_value)[2:]
                suggest_b = '0' * (const_width - len(suggest_b)) + suggest_b
                msg = f'''byte constant {token} may be confused for 4-bit nibble

Please either explicitly pad byte values smaller than 16 (e.g. {suggest_a}) or
write nibble-sized constants in bit form (e.g. {suggest_b}.)'''
                raise SyntaxWarning(msg)
            if spec_width % 8 != 0:
                suggest = bin(const_value)[2:]
                suggest = '0' * (const_width - len(suggest)) + suggest
                msg = f'''byte constant {token} is not byte-aligned

Check declaration for bugs, or re-write constant as {suggest} bits.'''
                raise IndexError(msg)
            fields.append(('c', spec_width, const_mask, const_value, const_width))
            spec_width += const_width
        elif ':' in token:
            var_name, var_kind = token.split(':')
            if not var_name.isidentifier() and var_name not in ('.', '-'):
                msg = f"{var_name} in {token} isn't a Python identifier or . or -"
                raise SyntaxError(msg)
            if is_signed := var_kind.startswith('s'):
                var_kind = var_kind[1:]
            var_width = int(var_kind)
            if var_name in ('.', '-'):
                const_value = 0
                const_mask = 0
                fields.append(('c', spec_width, const_mask, const_value, var_width))
            else:
                var_mask = (1 << var_width) - 1
                fields.append(('v', spec_width, var_mask, var_name, is_signed))
            spec_width += var_width
        else:
            # could add a opc2c-style name-length-encoding here, but it's more
            # cutesy code golf than a real space saver
            raise SyntaxError(f'''unidentified field {token!r}

Valid fields should be bytes (0xf00f) bits (010..11. where . is a "don't-care"
wildcard) or variable bindings (signed_jump:s24, imm:16, etc.)''')

    if spec_width % 8 != 0:
        msg = f'''{spec_width}-bit pattern width isn't a multiple of 8
        
Most encodings are byte-aligned. There might be a subtle error in the
specification, or just some forgotten '....' don't-care padding.'''
        raise IndexError(msg)

    # accumulate constant fields (remember that index is wrong bit-endianness)
    const_bits = [(mask, const, spec_width - index - width)
                  for ty,index,mask,const,width in fields if ty == 'c']
    mask, const = 0, 0
    for m,c,shift in const_bits:
        mask |= m << shift
        const |= c << shift
    match = Match(mask, const, spec_width // 8)

    # accumulate variable fields
    var_fields = [(name,Slice(mask << (spec_width - index - mask.bit_length()),
                             spec_width - index - mask.bit_length(),
                             signed,
                             spec_width // 8))
                  for ty,index,mask,name,signed in fields if ty == 'v']

    assert len(var_fields) == len(set(v[0] for v in var_fields)) # TODO dups
    var_fields = {name:slicer for name,slicer in var_fields}

    return match, var_fields

class Match(typing.NamedTuple):
    """Match some constant bits inside an exact length of bytes."""
    mask: int
    const: int
    byte_length: int
    def __repr__(self): 
        fmt = f'{{0:0{self.byte_length * 8}b}}'
        lut = {('0','0'):'.', ('0','1'):'.',
               ('1','0'):'0', ('1','1'):'1'}
        wildcard = ''.join([lut[w] for w in zip(fmt.format(self.mask),
                                                fmt.format(self.const))])
        return f'<Match {wildcard} /{self.byte_length}>'

    def matches(self, bytes):  # slow path
        """Exact-length pattern match.

        >>> m = bitspec.Match(0xffff00, 0xdead00, 4)
        >>> [m.matches(bs) for bs in [b'\\xff\\xde\\xad\\xff', 
        ...                           b'\\xff\\xde\\xad\\xffAAA']]
        [True, False]
        """
        if len(bytes) != self.byte_length:
            return False
        n = int.from_bytes(bytes, byteorder='big')
        return (n & self.mask) == self.const

    def expand_by(self, byte_length):
        """Don't change pattern, but add some don't-care bytes after it."""
        if byte_length == 0:
            return self
        elif byte_length > 0:
            bits = byte_length * 8
            return Match(self.mask << bits, self.const << bits, 
                         self.byte_length + byte_length)
        else:
            raise NotImplementedError(f'{byte_length} on {self}')

    @staticmethod
    def multimatch_precompute(matches) -> object:
        """Match lots of stuff at slightly better than O(n)
        
        [(Match, any)] -> bytes -> [any]
        """
        unique_masks = set((m.byte_length, m.mask) for m,_ in matches)
        first = lambda t:t[0]
        lut = {}
        for _, mask in unique_masks:
            const_lut = []
            for match,result in matches:
                const_lut += [(m.const, result) for m,result in matches
                                                if m.mask == mask]
            lut[mask] = {const: list(t[1] for t in group)
                         for const,group in groupby(sorted(const_lut, key=first),
                                                    key=first)}
        return unique_masks,lut

    @staticmethod
    def multimatch_execute(opaque, bytes) -> list:
        unique_masks,lut = opaque
        acc = []
        for byte_length, mask in unique_masks:
            if len(bytes) < byte_length:
                continue
            n = int.from_bytes(bytes[:byte_length], byteorder='big')
            acc += lut[mask].get(n & mask, [])
        return acc


class Slice(typing.NamedTuple):
    """Extract a contiguous region of bits from fixed length of bytes.

    >>> s = bitspec.Slice(0x00ff00, 8, True, 3)
    >>> [s.from_bytes(bs) for bs in [b'\\xfa\\xfe\\xaf', b'\\xaf\\xff\\xfa',
    ...                              b'\\xaf\\x00\\xfa', b'\\xfa\\x01\\xaf',
    ...                              b'A ABBB']]
    [-2, -1, 0, 1, 32]
    >>>
    """
    mask: int
    shift: int
    signed: bool
    byte_length: int

    def __repr__(self): 
        fmt = f'{{0:0{self.byte_length * 8}b}}'
        val = 'i' if self.signed else 'u'
        wildcard = fmt.format(self.mask).replace('1',val)
        if self.shift:  # s[:-0] == '', so 0 shift breaks printing
            wildcard = wildcard[:-self.shift] + '_'*self.shift
        return f'<Slice {rle(wildcard)} /{self.byte_length}>'

    def from_bytes(self, bs):  # slow path
        """Extract value from byte_length-sized prefix of bs.

        Raises:
            EOFError: not enough bytes
        """
        if len(bs) < self.byte_length:
            raise EOFError
        val = int.from_bytes(bs[:self.byte_length], byteorder='big') 
        val &= self.mask 
        val >>= self.shift
        if self.signed:
            width = (self.mask >> self.shift).bit_length()
            sign_mask = 1 << (width - 1)
            if val & sign_mask:
                val ^= sign_mask
                val -= sign_mask
        return val


def rle(s): return ' '.join(f'{c}:{len(list(g))}' for c, g in groupby(s))


def check_duplicate_args(var_fields, const_fields):
    sliced = var_fields.keys()
    const = const_fields.keys()
    if collision := set(sliced).intersection(set(const)):
        msg = f'''{collision} is both bit-sliced and assigned a constant value

If it's constant but not needed to disambiguate the pattern match, replace the
variable match with don't-care "." bits. Otherwise remove the {collision}
keyword argument while keeping the specification the same.'''
        raise NameError(msg)


def check_class_args(cls, var_fields, const_fields, specification):
    """Check if class constructor has matching arguments.
    
    This has to happen inside the decorator's closure in order to get access to
    the actual class object. As a result the stack trace no longer references
    the exact line that caused error, so print it manually.
    """
    # maybe check if dataclass fields also match, then don't implement
    # assembler if mismatching?
    if not (constructor := getattr(cls, '__init__')):
        msg = f'{cls} has no __init__ method, cannot validate bitspec variables'
        raise RuntimeError(msg)

    argspec = inspect.getfullargspec(constructor)
    if argspec.varargs != None or argspec.varkw != None:
        return  # we can't reason about *args or **kwargs, assume correctness

    fields = set(list(var_fields.keys()) + list(const_fields.keys()))

    if argspec.defaults == None:
        unfilled_args = set(argspec.args)
    else:
        unfilled_args = set(argspec.args[:-len(argspec.defaults)])
    unfilled_args.remove('self')
    if argspec.kwonlydefaults:
        unfilled_args -= set(argspec.kwonlydefaults.keys())
    unfilled_args -= fields

    if extra_fields := fields - set(argspec.args):
        msg_top = f"{extra_fields!r} not present in {cls.__name__}.__init__"
    elif unfilled_args:
        msg_top = f'constructor arguments {unfilled_args!r} not initialized'
    else:
        return  # best-effort "static" checks all ok

    msg = msg_top + f'''

This would cause an exception when from_bytes is called.
Specification: {specification}
Bit-sliced args: {list(var_fields.keys())}
Additional args: {list(const_fields.keys())}
Expected __init__ args: {inspect.signature(constructor)}'''
    raise NameError(msg)


def install_methods(cls, match, var_fields, const_fields):
    fields = var_fields; fields.update(const_fields)  
    # CAUTION: this mutated var_fields, currently this function consumes 'em
    #          but that may change in the future :\
    # Alternatives are {**a,**b} (3.5+) or a|b (3.9+)

    spec = cls.__dict__.get(_SPEC, {})
    if match in spec:
        msg = f'''pattern {match} is ambiguous
        
First option: {spec[match]!r}
Second option: {fields!r}'''
        raise SyntaxError(msg)
    if match != None:
        # Sometimes we decorate a class that can't be matched but anchors other
        # matches. Still set _SPEC so is_bitspec works.
        spec[match] = fields
    setattr(cls, _SPEC, spec)

    if not getattr(cls, 'from_bytes', None):
        setattr(cls, 'from_bytes', classmethod(from_bytes))
    if not getattr(cls, 'iter_bytes', None):
        setattr(cls, 'iter_bytes', classmethod(iter_bytes))
    set_new_attr(cls, '__len__', byte_length)
    set_new_attr(cls, 'to_bytes', to_bytes)
    return cls


def set_new_attr(cls, name, value):
    if name in cls.__dict__:
        assert getattr(cls, name) is value  # TODO allow overrides? how?
        return  # has already been set, don't need to do multiple times

    if isinstance(value, FunctionType):
        # adjust generated code (incorrect for statics)  (TODO move to codegen)
        value.__qualname__ = f'{cls.__qualname__}.{value.__name__}'

    setattr(cls, name, value)


# define declaration @ top of file
Bitspec.from_bytes = (from_bytes)
Bitspec.iter_bytes = (iter_bytes)
Bitspec.__len__ = byte_length
Bitspec.to_bytes = to_bytes


if __name__ == '__main__':
    import doctest
    import bitspec as bs
    doctest.testmod(
        #optionflags=doctest.REPORT_ONLY_FIRST_FAILURE,
        optionflags=doctest.ELLIPSIS,
        #verbose=True,
        globs={
            'bitspec':bs,
        },
    )
