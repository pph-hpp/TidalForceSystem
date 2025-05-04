#pragma once
#include <memory>
#include <assert.h>
namespace Render{

template<typename T>
class Singleton
{
public:
    static std::shared_ptr<T> GetInstance()
    {
        static std::shared_ptr<T> _instance = std::make_shared<T>();
        return _instance;
    }

protected:
    Singleton() = default;
    virtual ~Singleton() = default;

    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator=(Singleton&&) = delete;
};

}