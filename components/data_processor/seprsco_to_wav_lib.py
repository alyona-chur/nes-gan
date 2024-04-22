"""This module contains methods for seprsco to wav conversion.
Translated from Python2 nesmdb module. Uses VGMPlayer's vgm2wav.
"""
import binascii
from collections import OrderedDict
from pathlib import Path
import pickle
import struct
import subprocess
import tempfile

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

from components.common.constants import MUSIC_RATE
from components.common.constants import VGM2WAV_PATH


# Convert hexadecimal to binary and vv
b2h = lambda x: binascii.hexlify(x).decode('ascii')
h2b = lambda x: binascii.unhexlify(x.encode('ascii'))

# Convert integer to little-endian unsigned bytes
i2lub = lambda x: struct.pack('<I', x)
i2lusb = lambda x: struct.pack('<H', x)

# Convert character code to bytes
c2b = lambda x: struct.pack('B', x)


def seprsco_to_exprsco(seprsco):
    rate, nsamps, score = seprsco

    score_len = score.shape[0]

    exprsco = np.zeros((score_len, 4, 3), dtype=np.uint8)

    exprsco[:, :, 0] = score

    exprsco[:, :, 1] = 15
    exprsco[:, 2, 1] = 0

    return (rate, nsamps, exprsco)


def exprsco_to_rawsco(exprsco, clock=1789773.):
    rate, nsamps, exprsco = exprsco

    m = exprsco[:, :3, 0]
    m_zero = np.where(m == 0)

    m = m.astype(np.float32)
    f = 440 * np.power(2, ((m - 69) / 12))

    f_p, f_tr = f[:, :2], f[:, 2:]

    t_p = np.round((clock / (16 * f_p)) - 1)
    t_tr = np.round((clock / (32 * f_tr)) - 1)
    t = np.concatenate([t_p, t_tr], axis=1)

    t = t.astype(np.uint16)
    t[m_zero] = 0
    th = np.right_shift(np.bitwise_and(t, 0b11100000000), 8)
    tl = np.bitwise_and(t, 0b00011111111)

    rawsco = np.zeros((exprsco.shape[0], 4, 4), dtype=np.uint8)
    rawsco[:, :, 2:] = exprsco[:, :, 1:]
    rawsco[:, :3, 0] = th
    rawsco[:, :3, 1] = tl
    rawsco[:, 3, 1:] = exprsco[:, 3, :]

    return (clock, rate, nsamps, rawsco)


def rawsco_to_ndf(rawsco):
    clock, rate, nsamps, score = rawsco

    if rate == MUSIC_RATE:
        ar = True
    else:
        ar = False

    max_i = score.shape[0]

    samp = 0
    t = 0.
    # ('apu', ch, func, func_val, natoms, offset)
    ndf = [
        ('clock', int(clock)),
        ('apu', 'ch', 'p1', 0, 0, 0),
        ('apu', 'ch', 'p2', 0, 0, 0),
        ('apu', 'ch', 'tr', 0, 0, 0),
        ('apu', 'ch', 'no', 0, 0, 0),
        ('apu', 'p1', 'du', 0, 1, 0),
        ('apu', 'p1', 'lh', 1, 1, 0),
        ('apu', 'p1', 'cv', 1, 1, 0),
        ('apu', 'p1', 'vo', 0, 1, 0),
        ('apu', 'p1', 'ss', 7, 2, 1),  # This is necessary to prevent channel silence for low notes
        ('apu', 'p2', 'du', 0, 3, 0),
        ('apu', 'p2', 'lh', 1, 3, 0),
        ('apu', 'p2', 'cv', 1, 3, 0),
        ('apu', 'p2', 'vo', 0, 3, 0),
        ('apu', 'p2', 'ss', 7, 4, 1),  # This is necessary to prevent channel silence for low notes
        ('apu', 'tr', 'lh', 1, 5, 0),
        ('apu', 'tr', 'lr', 127, 5, 0),
        ('apu', 'no', 'lh', 1, 6, 0),
        ('apu', 'no', 'cv', 1, 6, 0),
        ('apu', 'no', 'vo', 0, 6, 0),
    ]
    ch_to_last_tl = {ch: 0 for ch in ['p1', 'p2']}
    ch_to_last_th = {ch: 0 for ch in ['p1', 'p2']}
    ch_to_last_timer = {ch: 0 for ch in ['p1', 'p2', 'tr']}
    ch_to_last_du = {ch: 0 for ch in ['p1', 'p2']}
    ch_to_last_volume = {ch: 0 for ch in ['p1', 'p2', 'no']}
    last_no_np = 0
    last_no_nl = 0

    for i in range(max_i):
        for j, ch in enumerate(['p1', 'p2']):
            th, tl, volume, du = score[i, j]
            timer = (th << 8) + tl
            last_timer = ch_to_last_timer[ch]

            # NOTE: This will never be perfect reconstruction because phase
            # is not incremented when the channel is off
            retrigger = False
            if last_timer == 0 and timer != 0:
                ndf.append(('apu', 'ch', ch, 1, 0, 0))
                retrigger = True
            elif last_timer != 0 and timer == 0:
                ndf.append(('apu', 'ch', ch, 0, 0, 0))

            if du != ch_to_last_du[ch]:
                ndf.append(('apu', ch, 'du', du, 0, 0))
                ch_to_last_du[ch] = du

            if volume > 0 and volume != ch_to_last_volume[ch]:
                ndf.append(('apu', ch, 'vo', volume, 0, 0))
            ch_to_last_volume[ch] = volume

            if tl != ch_to_last_tl[ch]:
                ndf.append(('apu', ch, 'tl', tl, 0, 2))
                ch_to_last_tl[ch] = tl
            if retrigger or th != ch_to_last_th[ch]:
                ndf.append(('apu', ch, 'th', th, 0, 3))
                ch_to_last_th[ch] = th

            ch_to_last_timer[ch] = timer

        j = 2
        ch = 'tr'
        th, tl, _, _ = score[i, j]
        timer = (th << 8) + tl
        last_timer = ch_to_last_timer[ch]
        if last_timer == 0 and timer != 0:
            ndf.append(('apu', 'ch', ch, 1, 0, 0))
        elif last_timer != 0 and timer == 0:
            ndf.append(('apu', 'ch', ch, 0, 0, 0))
        if timer != last_timer:
            ndf.append(('apu', ch, 'tl', tl, 0, 2))
            ndf.append(('apu', ch, 'th', th, 0, 3))
        ch_to_last_timer[ch] = timer

        j = 3
        ch = 'no'
        _, np, volume, nl = score[i, j]
        if last_no_np == 0 and np != 0:
            ndf.append(('apu', 'ch', ch, 1, 0, 0))
        elif last_no_np != 0 and np == 0:
            ndf.append(('apu', 'ch', ch, 0, 0, 0))
        if volume > 0 and volume != ch_to_last_volume[ch]:
            ndf.append(('apu', ch, 'vo', volume, 0, 0))
        ch_to_last_volume[ch] = volume
        if nl != last_no_nl:
            ndf.append(('apu', ch, 'nl', nl, 0, 2))
            last_no_nl = nl
        if np > 0 and np != last_no_np:
            ndf.append(('apu', ch, 'np', 16 - np, 0, 2))
            ndf.append(('apu', ch, 'll', 0, 0, 3))
        last_no_np = np

        if ar:
            wait_amt = 1
        else:
            fs = MUSIC_RATE * 1.0
            t += 1. / rate
            wait_amt = min(int(fs * t) - samp, nsamps - samp)

        ndf.append(('wait', wait_amt))
        samp += wait_amt

    remaining = nsamps - samp
    assert remaining >= 0
    if remaining > 0:
        ndf.append(('wait', remaining))

    return ndf


def get_register_function_bitmasks():
    register_function_bitmasks = {
        # P1/P2
        # du: duty
        # lh: length counter halt
        # cv: constant volume
        # vo: volume
        # se: sweep enable
        # sp: sweep period
        # sn: sweep negation
        # ss: sweep shift
        # tl: timer low
        # ll: length counter load
        # th: timer hi
        'p1': {
            0: [
                ('du', 0b11000000),
                ('lh', 0b00100000),
                ('cv', 0b00010000),
                ('vo', 0b00001111)
            ],
            1: [
                ('se', 0b10000000),
                ('sp', 0b01110000),
                ('sn', 0b00001000),
                ('ss', 0b00000111)
            ],
            2: [
                ('tl', 0b11111111)
            ],
            3: [
                ('ll', 0b11111000),
                ('th', 0b00000111)
            ]
        },
        # Triangle
        # lh: length counter halt / linear counter control
        # lr: linear counter load
        # tl: timer lo
        # ll: length counter load
        # th: timer hi
        'tr': {
            0: [
                ('lh', 0b10000000),
                ('lr', 0b01111111)
            ],
            1: [],
            2: [
                ('tl', 0b11111111)
            ],
            3: [
                ('ll', 0b11111000),
                ('th', 0b00000111)
            ]
        },
        # Noise
        # lh: length counter halt / envelope loop
        # cv: constant volume
        # vo: volume
        # nl: noise loop
        # np: noise period
        # ll: length counter load
        'no': {
            0: [
                ('lh', 0b00100000),
                ('cv', 0b00010000),
                ('vo', 0b00001111)
            ],
            1: [],
            2: [
                ('nl', 0b10000000),
                ('np', 0b00001111)
            ],
            3: [
                ('ll', 0b11111000)
            ]
        },
        # DMC
        # iq: irq enable
        # lo: loop
        # fr: frequency
        # lc: load counter
        # sa: sample address
        # sl: sample length
        'dm': {
            0: [
                ('iq', 0b10000000),
                ('lo', 0b01000000),
                ('fr', 0b00001111)
            ],
            1: [
                ('lc', 0b01111111)
            ],
            2: [
                ('sa', 0b11111111)
            ],
            3: [
                ('sl', 0b11111111)
            ]
        },
        # Channel status
        # dm: enable dmc
        # no: enable noise
        # tr: enable triangle
        # p2: enable p2
        # p1: enable p1
        'ch': {
            0: [
                ('dm', 0b00010000),
                ('no', 0b00001000),
                ('tr', 0b00000100),
                ('p2', 0b00000010),
                ('p1', 0b00000001)
            ]
        },
        # Frame counter
        # mo: mode
        # iq: irq inhibit flag
        'fc': {
            0: [
                ('mo', 0b10000000),
                ('iq', 0b01000000)
            ]
        }
    }
    register_function_bitmasks['p2'] = register_function_bitmasks['p1']
    return register_function_bitmasks


def func_to_bitmask(ch, fu):
    register_function_bitmasks = get_register_function_bitmasks()
    for _, bitmasks in register_function_bitmasks[ch].items():
        for fu_name, bitmask in bitmasks:
            if fu_name == fu:
                return bitmask
    raise ValueError()


def get_register_memory_offsets():
    return {
        'p1': 0x00,
        'p2': 0x04,
        'tr': 0x08,
        'no': 0x0c,
        'dm': 0x10,
        'ch': 0x15,
        'fc': 0x17
    }


def ndf_to_ndr(ndf):
    ndr = ndf[:1]
    ndf = ndf[1:]

    registers = {
        'p1': [0x00] * 4,
        'p2': [0x00] * 4,
        'tr': [0x00] * 4,
        'no': [0x00] * 4,
        'dm': [0x00] * 4,
        'ch': [0x00],
        'fc': [0x00]
    }

    # Convert commands to VGM
    regn_to_val = OrderedDict()
    for comm in ndf:
        itype = comm[0]
        if itype == 'wait':
            for _, (arg1, arg2) in regn_to_val.items():
                ndr.append(('apu', b2h(c2b(arg1)), b2h(c2b(arg2))))
            regn_to_val = OrderedDict()

            amt = comm[1]

            ndr.append(('wait', amt))
        elif itype == 'apu':
            dest = comm[1]
            param = comm[2]
            val = comm[3]
            natoms = comm[4]
            param_offset = comm[5]

            # Find offset/bitmask
            reg = registers[dest]
            param_bitmask = func_to_bitmask(dest, param)

            # Apply mask
            mask_bin = '{:08b}'.format(param_bitmask)
            nbits = mask_bin.count('1')
            if val < 0 or val >= (2 ** nbits):
                raise ValueError('{}, {} (0, {}]: invalid value specified {}'.format(
                    comm[1], comm[2], (2 ** nbits), val))
            assert val >= 0 and val < (2 ** nbits)
            shift = max(0, 7 - mask_bin.rfind('1')) % 8
            val_old = reg[param_offset]
            reg[param_offset] &= (255 - param_bitmask)
            reg[param_offset] |= val << shift
            assert reg[param_offset] < 256
            val_new = reg[param_offset]

            arg1 = get_register_memory_offsets()[dest] + param_offset
            arg2 = reg[param_offset]

            regn_to_val[(dest, param_offset, natoms)] = (arg1, arg2)
        elif itype == 'ram':
            # TODO
            continue
        else:
            raise NotImplementedError()

    for _, (arg1, arg2) in regn_to_val.items():
        ndr.append(('apu', b2h(c2b(arg1)), b2h(c2b(arg2))))

    return ndr


def ndr_to_vgm(ndr):
    assert ndr[0][0] == 'clock', "First element must define the clock rate"
    clock = ndr[0][1]

    ndr = ndr[1:]

    EMPTYBYTE = i2lub(0)
    flatten = lambda data: [item for sublist in data for item in sublist]

    # Initialize the VGM list with 48 empty bytes
    vgm = bytearray(flatten([EMPTYBYTE] * 48))

    # Set VGM identifier and version
    vgm[:4] = bytearray([0x56, 0x67, 0x6d, 0x20])  # 'Vgm '
    vgm[8:12] = i2lub(0x161)
    # Set clock rate
    vgm[132:136] = i2lub(clock)
    # Set data offset
    vgm[52:56] = i2lub(0xc0 - 0x34)

    wait_sum = 0
    for comm in ndr:
        itype = comm[0]
        if itype == 'wait':
            amt = comm[1]
            wait_sum += amt

            while amt > 65535:
                vgm.extend(c2b(0x61))
                vgm.extend(i2lusb(65535))
                amt -= 65535

            vgm.extend(c2b(0x61))
            vgm.extend(i2lusb(amt))
        elif itype == 'apu':
            arg1 = h2b(comm[1])
            arg2 = h2b(comm[2])
            vgm.extend(c2b(0xb4))
            vgm.extend(arg1)
            vgm.extend(arg2)
        elif itype == 'ram':
            raise NotImplementedError("RAM operations not implemented")
        else:
            raise NotImplementedError(f"Command type {itype} not implemented")

    # Append halt command
    vgm.append(0x66)

    # Set total samples and EoF offset
    vgm[24:28] = i2lub(wait_sum)
    vgm[4:8] = i2lub(len(vgm) - 4)

    return bytes(vgm)


def load_vgmwav(wav_fp):
    fs, wav = wavread(wav_fp)
    assert fs == MUSIC_RATE
    if wav.ndim == 2:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    wav /= 32767.
    return wav


def vgm_to_wav(vgm):
    bin_fp = VGM2WAV_PATH

    vf = tempfile.NamedTemporaryFile('wb')
    wf = tempfile.NamedTemporaryFile('rb')

    vf.write(vgm)
    vf.seek(0)

    res = subprocess.call('{} --loop-count 1 {} {}'.format(bin_fp, vf.name, wf.name).split())
    if res > 0:
        vf.close()
        wf.close()
        raise Exception('Invalid return code {} from vgm2wav'.format(res))

    vf.close()

    wf.seek(0)
    wav = load_vgmwav(wf.name)

    wf.close()

    return wav


def seprsco_to_wav(seprsco):
    exprsco = seprsco_to_exprsco(seprsco)
    rawsco = exprsco_to_rawsco(exprsco)
    ndf = rawsco_to_ndf(rawsco)  # NES Music Database Format
    ndr = ndf_to_ndr(ndf)
    vgm = ndr_to_vgm(ndr)
    wav = vgm_to_wav(vgm)
    return wav


def convert_seprsco_to_wav(pkl_file: Path, wav_dir: Path):
    """Converts pkl file with seprsco format song rto wav file."""
    seprsco = None
    with open(pkl_file, 'rb') as f:
        seprsco = pickle.load(f)
    wav = seprsco_to_wav(seprsco)
    wav *= 32767.
    wav = np.clip(wav, -32767., 32767.)
    wav = wav.astype(np.int16)

    wav_file = str(wav_dir / f'{pkl_file.stem}.wav')
    wavwrite(str(wav_file), MUSIC_RATE, wav)  # 44.1 kHz timing resolution
