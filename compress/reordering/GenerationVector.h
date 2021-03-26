#ifndef EFANNA2E_GENERATIONVECTOR_H
#define EFANNA2E_GENERATIONVECTOR_H

#include <vector>

namespace compress {

template<typename ValT, typename GenT>
class GenerationVector {
private:
  class Entry {
    friend class GenerationVector;
  public:
    Entry() : _value(), _generation(0) {}

    bool hasValue(GenT generation) const { return _generation == generation; }

    void setValue(GenT generation, ValT v) {
      _value = v;
      _generation = generation;
    }
  private:
    ValT _value;
    GenT _generation;
  };

public:
  const ValT& operator[](std::size_t i) const {
    const auto& entry = _data[i];
    return entry.hasValue(_generation) ? entry._value : _defaultValue;
  }

  bool getValue(std::size_t i, ValT* val) const {
    const auto& entry = _data[i];
    if (entry.hasValue(_generation)) {
      *val = entry._value;
      return true;
    }
    return false;
  }

  bool hasValue(std::size_t i) const { return _data[i].hasValue(_generation); }

  void setValue(std::size_t i, ValT v) {
    _data[i].setValue(_generation, v);
  }

  void clearOrInit(std::size_t size) {
    ++_generation;
    if (size > _data.size()) {
      _data.resize(size);
    }
  }

  std::size_t size() const { return _data.size(); }

private:
  std::vector<Entry> _data;
  GenT _generation = 0;
  ValT _defaultValue = 0;
};

}


#endif //EFANNA2E_GENERATIONVECTOR_H
