#include <vector>

class ByteBufferBuilder {
public:
    std::vector<std::uint8_t> data;

    template<typename T>
    void append(const T* ptr, size_t count = 1);

    void clear() { data.clear(); }

    void* raw() { return data.data(); }
    size_t size() const { return data.size(); }
};


template<typename T>
void ByteBufferBuilder::append(const T* ptr, size_t count) {
    size_t bytes = sizeof(T) * count;
    size_t offset = data.size();
    data.resize(offset + bytes);
    std::memcpy(data.data() + offset, ptr, bytes);
}