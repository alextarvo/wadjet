# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cut_images.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='cut_images.proto',
  package='wadjet',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10\x63ut_images.proto\x12\x06wadjet\"2\n\x0f\x42\x61llCoordinates\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x12\t\n\x01r\x18\x03 \x01(\x05\"u\n\x08\x43utImage\x12,\n\x0b\x63oordinates\x18\x01 \x01(\x0b\x32\x17.wadjet.BallCoordinates\x12\r\n\x05image\x18\n \x01(\x0c\x12\x0c\n\x04rows\x18\x0b \x01(\x05\x12\x0c\n\x04\x63ols\x18\x0c \x01(\x05\x12\x10\n\x08\x63hannels\x18\r \x01(\x05\"3\n\x0b\x43utImageSet\x12$\n\ncut_images\x18\x01 \x03(\x0b\x32\x10.wadjet.CutImage'
)




_BALLCOORDINATES = _descriptor.Descriptor(
  name='BallCoordinates',
  full_name='wadjet.BallCoordinates',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='wadjet.BallCoordinates.x', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='wadjet.BallCoordinates.y', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='r', full_name='wadjet.BallCoordinates.r', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=78,
)


_CUTIMAGE = _descriptor.Descriptor(
  name='CutImage',
  full_name='wadjet.CutImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='coordinates', full_name='wadjet.CutImage.coordinates', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image', full_name='wadjet.CutImage.image', index=1,
      number=10, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rows', full_name='wadjet.CutImage.rows', index=2,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cols', full_name='wadjet.CutImage.cols', index=3,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='channels', full_name='wadjet.CutImage.channels', index=4,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=80,
  serialized_end=197,
)


_CUTIMAGESET = _descriptor.Descriptor(
  name='CutImageSet',
  full_name='wadjet.CutImageSet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='cut_images', full_name='wadjet.CutImageSet.cut_images', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=199,
  serialized_end=250,
)

_CUTIMAGE.fields_by_name['coordinates'].message_type = _BALLCOORDINATES
_CUTIMAGESET.fields_by_name['cut_images'].message_type = _CUTIMAGE
DESCRIPTOR.message_types_by_name['BallCoordinates'] = _BALLCOORDINATES
DESCRIPTOR.message_types_by_name['CutImage'] = _CUTIMAGE
DESCRIPTOR.message_types_by_name['CutImageSet'] = _CUTIMAGESET
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BallCoordinates = _reflection.GeneratedProtocolMessageType('BallCoordinates', (_message.Message,), {
  'DESCRIPTOR' : _BALLCOORDINATES,
  '__module__' : 'cut_images_pb2'
  # @@protoc_insertion_point(class_scope:wadjet.BallCoordinates)
  })
_sym_db.RegisterMessage(BallCoordinates)

CutImage = _reflection.GeneratedProtocolMessageType('CutImage', (_message.Message,), {
  'DESCRIPTOR' : _CUTIMAGE,
  '__module__' : 'cut_images_pb2'
  # @@protoc_insertion_point(class_scope:wadjet.CutImage)
  })
_sym_db.RegisterMessage(CutImage)

CutImageSet = _reflection.GeneratedProtocolMessageType('CutImageSet', (_message.Message,), {
  'DESCRIPTOR' : _CUTIMAGESET,
  '__module__' : 'cut_images_pb2'
  # @@protoc_insertion_point(class_scope:wadjet.CutImageSet)
  })
_sym_db.RegisterMessage(CutImageSet)


# @@protoc_insertion_point(module_scope)
