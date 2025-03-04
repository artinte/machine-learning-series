#include <iostream>
#include <memory>
#include <cstring>

class Person {
    public:
        // 构造函数
        Person(const char* name, int age): age(age) {
            this->name = new char[strlen(name) + 1];
            strcpy(this->name, name);
            count++;
        }

        // 拷贝构造函数（深拷贝）
        Person(const Person& other) : age(other.age) {
            name = new char[strlen(other.name) + 1];
            strcpy(name, other.name);
        }

        // 移动构造函数
        Person(Person&& other) : name(other.name), age(other.age) {
            other.name = nullptr;
        }

        // 拷贝赋值运算符（深拷贝）
        Person& operator=(const Person& other) {
            if (this == &other)
                return *this;
            delete[] name;
            name = new char[strlen(other.name) + 1];
            strcpy(name, other.name);
            age = other.age;
            return *this;
        }

        // 移动赋值运算符
        Person& operator=(Person&& other) {
            if (this == &other)
                return *this;
            delete[] name;
            name = other.name;
            age = other.age;
            other.name = nullptr;
            return *this;
        }

        // 析构函数
        ~Person() {
            delete[] name;
            count--;
        }

        void show() const {
            std::cout << "Name: " << name << ", age: " << age << std::endl;
        }
    private:
        char* name;
        int age;
        static int count;
};

int Person::count = 0;

int main(int argc, char* argv[]) {
    Person p1("Alice", 25);
    p1.show();
    // 调用拷贝构造函数
    Person p2 = p1;
    p2.show();

    Person p3("Bob", 30);
    p3.show();

    // 调用拷贝赋值运算符
    p3 = p1;

    // 调用移动构造函数
    Person p4 = std::move(p3);
    p4.show();
}