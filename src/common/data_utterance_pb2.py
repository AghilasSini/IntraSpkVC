# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: data_utterance.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14\x64\x61ta_utterance.proto\x12\x0e\x64\x61ta_utterance\"B\n\x0c\x44oubleMatrix\x12\x0f\n\x07num_row\x18\x01 \x01(\r\x12\x0f\n\x07num_col\x18\x02 \x01(\r\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x01\x42\x02\x10\x01\"A\n\x0b\x46loatMatrix\x12\x0f\n\x07num_row\x18\x01 \x01(\r\x12\x0f\n\x07num_col\x18\x02 \x01(\r\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x02\x42\x02\x10\x01\"A\n\x0bInt32Matrix\x12\x0f\n\x07num_row\x18\x01 \x01(\r\x12\x0f\n\x07num_col\x18\x02 \x01(\r\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x05\x42\x02\x10\x01\"B\n\x0c\x42inaryMatrix\x12\x0f\n\x07num_row\x18\x01 \x01(\r\x12\x0f\n\x07num_col\x18\x02 \x01(\r\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x08\x42\x02\x10\x01\"\x8b\x01\n\x07Segment\x12\x0e\n\x06symbol\x18\x01 \x03(\t\x12/\n\nstart_time\x18\x02 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12-\n\x08\x65nd_time\x18\x03 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12\x10\n\x08num_item\x18\x04 \x01(\x05\"\xd4\x07\n\x0eVocoderFeature\x12;\n\x07vocoder\x18\x01 \x01(\x0e\x32*.data_utterance.VocoderFeature.VocoderName\x12\x35\n\x06\x66ilter\x18\x02 \x01(\x0b\x32%.data_utterance.VocoderFeature.Filter\x12\x35\n\x06source\x18\x03 \x01(\x0b\x32%.data_utterance.VocoderFeature.Source\x12;\n\x05param\x18\x04 \x01(\x0b\x32,.data_utterance.VocoderFeature.AnalysisParam\x1a\x89\x01\n\x06\x46ilter\x12)\n\x04spec\x18\x01 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12)\n\x04mfcc\x18\x02 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12)\n\x04mcep\x18\x03 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x1a\xe6\x01\n\x06Source\x12\'\n\x02\x66\x30\x18\x01 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12\'\n\x02\x61p\x18\x02 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12(\n\x03\x62\x61p\x18\x03 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12(\n\x03vuv\x18\x04 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12\x36\n\x11temporal_position\x18\x05 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x1a\xa0\x02\n\rAnalysisParam\x12\x13\n\x0bwindow_size\x18\x01 \x01(\x02\x12\x13\n\x0bwindow_type\x18\x02 \x01(\t\x12\r\n\x05shift\x18\x03 \x01(\x02\x12\x11\n\tnum_frame\x18\x04 \x01(\x05\x12\r\n\x05\x61lpha\x18\x05 \x01(\x02\x12\x10\n\x08\x66\x66t_size\x18\x06 \x01(\x05\x12\x10\n\x08spec_dim\x18\x07 \x01(\x05\x12\x10\n\x08mfcc_dim\x18\x08 \x01(\x05\x12\x10\n\x08mcep_dim\x18\t \x01(\x05\x12\x10\n\x08\x66\x30_floor\x18\n \x01(\x02\x12\x0f\n\x07\x66\x30_ceil\x18\x0b \x01(\x02\x12\x11\n\ttimestamp\x18\x0c \x01(\t\x12\x0e\n\x06\x61p_dim\x18\r \x01(\x05\x12\x0f\n\x07\x62\x61p_dim\x18\x0e \x01(\x05\x12\x15\n\rpitch_tracker\x18\x0f \x01(\t\"B\n\x0bVocoderName\x12\t\n\x05WORLD\x10\x00\x12\x13\n\x0fTANDEM_STRAIGHT\x10\x01\x12\x13\n\x0fLEGACY_STRAIGHT\x10\x02\"\xd2\x02\n\x08MetaData\x12\x12\n\nspeaker_id\x18\x01 \x01(\t\x12\x31\n\x07\x64ialect\x18\x02 \x01(\x0e\x32 .data_utterance.MetaData.Dialect\x12/\n\x06gender\x18\x03 \x01(\x0e\x32\x1f.data_utterance.MetaData.Gender\x12\x15\n\roriginal_file\x18\x04 \x01(\t\x12\x13\n\x0bnum_channel\x18\x05 \x01(\x05\"\x82\x01\n\x07\x44ialect\x12\t\n\x05\x45N_US\x10\x00\x12\t\n\x05\x45N_CN\x10\x01\x12\t\n\x05\x45N_SP\x10\x02\x12\t\n\x05\x45N_ES\x10\x03\x12\t\n\x05\x45N_AB\x10\x04\x12\t\n\x05\x45N_KR\x10\x05\x12\t\n\x05\x45N_IN\x10\x06\x12\t\n\x05\x45N_VN\x10\x07\x12\t\n\x05\x45N_CA\x10\x08\x12\t\n\x05\x45N_GB\x10\t\x12\t\n\x05\x45N_XS\x10\n\"\x1d\n\x06Gender\x12\x05\n\x01M\x10\x00\x12\x05\n\x01\x46\x10\x01\x12\x05\n\x01O\x10\x02\"E\n\nKaldiParam\x12\r\n\x05shift\x18\x01 \x01(\x02\x12\x13\n\x0bwindow_size\x18\x02 \x01(\x02\x12\x13\n\x0bwindow_type\x18\x03 \x01(\t\"\xea\x03\n\rDataUtterance\x12(\n\x03wav\x18\x01 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12\n\n\x02\x66s\x18\x02 \x01(\x05\x12\x0c\n\x04text\x18\x03 \x01(\t\x12\r\n\x05\x61lign\x18\x04 \x01(\t\x12(\n\x03ppg\x18\x05 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12\x32\n\rmonophone_ppg\x18\x06 \x01(\x0b\x32\x1b.data_utterance.FloatMatrix\x12&\n\x05phone\x18\x07 \x01(\x0b\x32\x17.data_utterance.Segment\x12%\n\x04word\x18\x08 \x01(\x0b\x32\x17.data_utterance.Segment\x12(\n\x03lab\x18\t \x01(\x0b\x32\x1b.data_utterance.Int32Matrix\x12\x34\n\x0cvocoder_feat\x18\n \x01(\x0b\x32\x1e.data_utterance.VocoderFeature\x12+\n\tmeta_data\x18\x0b \x01(\x0b\x32\x18.data_utterance.MetaData\x12\x14\n\x0cutterance_id\x18\x0c \x01(\t\x12/\n\x0bkaldi_param\x18\r \x01(\x0b\x32\x1a.data_utterance.KaldiParam*\x05\x08\x65\x10\xc9\x01')



_DOUBLEMATRIX = DESCRIPTOR.message_types_by_name['DoubleMatrix']
_FLOATMATRIX = DESCRIPTOR.message_types_by_name['FloatMatrix']
_INT32MATRIX = DESCRIPTOR.message_types_by_name['Int32Matrix']
_BINARYMATRIX = DESCRIPTOR.message_types_by_name['BinaryMatrix']
_SEGMENT = DESCRIPTOR.message_types_by_name['Segment']
_VOCODERFEATURE = DESCRIPTOR.message_types_by_name['VocoderFeature']
_VOCODERFEATURE_FILTER = _VOCODERFEATURE.nested_types_by_name['Filter']
_VOCODERFEATURE_SOURCE = _VOCODERFEATURE.nested_types_by_name['Source']
_VOCODERFEATURE_ANALYSISPARAM = _VOCODERFEATURE.nested_types_by_name['AnalysisParam']
_METADATA = DESCRIPTOR.message_types_by_name['MetaData']
_KALDIPARAM = DESCRIPTOR.message_types_by_name['KaldiParam']
_DATAUTTERANCE = DESCRIPTOR.message_types_by_name['DataUtterance']
_VOCODERFEATURE_VOCODERNAME = _VOCODERFEATURE.enum_types_by_name['VocoderName']
_METADATA_DIALECT = _METADATA.enum_types_by_name['Dialect']
_METADATA_GENDER = _METADATA.enum_types_by_name['Gender']
DoubleMatrix = _reflection.GeneratedProtocolMessageType('DoubleMatrix', (_message.Message,), {
  'DESCRIPTOR' : _DOUBLEMATRIX,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.DoubleMatrix)
  })
_sym_db.RegisterMessage(DoubleMatrix)

FloatMatrix = _reflection.GeneratedProtocolMessageType('FloatMatrix', (_message.Message,), {
  'DESCRIPTOR' : _FLOATMATRIX,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.FloatMatrix)
  })
_sym_db.RegisterMessage(FloatMatrix)

Int32Matrix = _reflection.GeneratedProtocolMessageType('Int32Matrix', (_message.Message,), {
  'DESCRIPTOR' : _INT32MATRIX,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.Int32Matrix)
  })
_sym_db.RegisterMessage(Int32Matrix)

BinaryMatrix = _reflection.GeneratedProtocolMessageType('BinaryMatrix', (_message.Message,), {
  'DESCRIPTOR' : _BINARYMATRIX,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.BinaryMatrix)
  })
_sym_db.RegisterMessage(BinaryMatrix)

Segment = _reflection.GeneratedProtocolMessageType('Segment', (_message.Message,), {
  'DESCRIPTOR' : _SEGMENT,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.Segment)
  })
_sym_db.RegisterMessage(Segment)

VocoderFeature = _reflection.GeneratedProtocolMessageType('VocoderFeature', (_message.Message,), {

  'Filter' : _reflection.GeneratedProtocolMessageType('Filter', (_message.Message,), {
    'DESCRIPTOR' : _VOCODERFEATURE_FILTER,
    '__module__' : 'data_utterance_pb2'
    # @@protoc_insertion_point(class_scope:data_utterance.VocoderFeature.Filter)
    })
  ,

  'Source' : _reflection.GeneratedProtocolMessageType('Source', (_message.Message,), {
    'DESCRIPTOR' : _VOCODERFEATURE_SOURCE,
    '__module__' : 'data_utterance_pb2'
    # @@protoc_insertion_point(class_scope:data_utterance.VocoderFeature.Source)
    })
  ,

  'AnalysisParam' : _reflection.GeneratedProtocolMessageType('AnalysisParam', (_message.Message,), {
    'DESCRIPTOR' : _VOCODERFEATURE_ANALYSISPARAM,
    '__module__' : 'data_utterance_pb2'
    # @@protoc_insertion_point(class_scope:data_utterance.VocoderFeature.AnalysisParam)
    })
  ,
  'DESCRIPTOR' : _VOCODERFEATURE,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.VocoderFeature)
  })
_sym_db.RegisterMessage(VocoderFeature)
_sym_db.RegisterMessage(VocoderFeature.Filter)
_sym_db.RegisterMessage(VocoderFeature.Source)
_sym_db.RegisterMessage(VocoderFeature.AnalysisParam)

MetaData = _reflection.GeneratedProtocolMessageType('MetaData', (_message.Message,), {
  'DESCRIPTOR' : _METADATA,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.MetaData)
  })
_sym_db.RegisterMessage(MetaData)

KaldiParam = _reflection.GeneratedProtocolMessageType('KaldiParam', (_message.Message,), {
  'DESCRIPTOR' : _KALDIPARAM,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.KaldiParam)
  })
_sym_db.RegisterMessage(KaldiParam)

DataUtterance = _reflection.GeneratedProtocolMessageType('DataUtterance', (_message.Message,), {
  'DESCRIPTOR' : _DATAUTTERANCE,
  '__module__' : 'data_utterance_pb2'
  # @@protoc_insertion_point(class_scope:data_utterance.DataUtterance)
  })
_sym_db.RegisterMessage(DataUtterance)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DOUBLEMATRIX.fields_by_name['data']._options = None
  _DOUBLEMATRIX.fields_by_name['data']._serialized_options = b'\020\001'
  _FLOATMATRIX.fields_by_name['data']._options = None
  _FLOATMATRIX.fields_by_name['data']._serialized_options = b'\020\001'
  _INT32MATRIX.fields_by_name['data']._options = None
  _INT32MATRIX.fields_by_name['data']._serialized_options = b'\020\001'
  _BINARYMATRIX.fields_by_name['data']._options = None
  _BINARYMATRIX.fields_by_name['data']._serialized_options = b'\020\001'
  _DOUBLEMATRIX._serialized_start=40
  _DOUBLEMATRIX._serialized_end=106
  _FLOATMATRIX._serialized_start=108
  _FLOATMATRIX._serialized_end=173
  _INT32MATRIX._serialized_start=175
  _INT32MATRIX._serialized_end=240
  _BINARYMATRIX._serialized_start=242
  _BINARYMATRIX._serialized_end=308
  _SEGMENT._serialized_start=311
  _SEGMENT._serialized_end=450
  _VOCODERFEATURE._serialized_start=453
  _VOCODERFEATURE._serialized_end=1433
  _VOCODERFEATURE_FILTER._serialized_start=704
  _VOCODERFEATURE_FILTER._serialized_end=841
  _VOCODERFEATURE_SOURCE._serialized_start=844
  _VOCODERFEATURE_SOURCE._serialized_end=1074
  _VOCODERFEATURE_ANALYSISPARAM._serialized_start=1077
  _VOCODERFEATURE_ANALYSISPARAM._serialized_end=1365
  _VOCODERFEATURE_VOCODERNAME._serialized_start=1367
  _VOCODERFEATURE_VOCODERNAME._serialized_end=1433
  _METADATA._serialized_start=1436
  _METADATA._serialized_end=1774
  _METADATA_DIALECT._serialized_start=1613
  _METADATA_DIALECT._serialized_end=1743
  _METADATA_GENDER._serialized_start=1745
  _METADATA_GENDER._serialized_end=1774
  _KALDIPARAM._serialized_start=1776
  _KALDIPARAM._serialized_end=1845
  _DATAUTTERANCE._serialized_start=1848
  _DATAUTTERANCE._serialized_end=2338
# @@protoc_insertion_point(module_scope)
