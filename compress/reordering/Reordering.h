#ifndef EFANNA2E_REORDERING_H
#define EFANNA2E_REORDERING_H

#include <compress/reordering/RecursiveGraphBisection.h>

namespace compress {

class Reordering {
public:
  static bool genReorderMapping(const std::vector<std::vector<label_t>>& graph,
                                std::vector<label_t>* mapping);

  static void reorder(std::size_t vecSize, const std::vector<label_t>& mapping,
                      std::vector<std::vector<label_t>>* graph,
                      std::vector<label_t>* eps,
                      uint8_t* vecs);
};

}


#endif //EFANNA2E_REORDERING_H
