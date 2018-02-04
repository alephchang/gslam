#pragma once
#include"3rdparty/DBoW2/DBoW2/FORB.h"
#include"3rdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

namespace gslam
{

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBVocabulary;

} //namespace 

