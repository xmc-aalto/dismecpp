// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_HYPERPARAMS_H
#define DISMEC_HYPERPARAMS_H

#include <string>
#include <variant>
#include <unordered_map>
#include <type_traits>
#include <functional>
#include <stdexcept>

namespace dismec
{
    /*! \class HyperParameterBase
     *  \brief Base class for all objects that have adjustable hyper-parameters
     *  \details This class defines a common interface for all objects with adjustable
     *  hyper-parameters, and provides the implementation. Derived classes only have to
     *  declare their hyper-parameters in the constructor using the `declare_hyper_parameter`
     *  functions, and the rest is handled by this class.
     *
     *  This is some fairly complicated C++ magic; however, if you only want to modify
     *  the solver algorithms or objective functions, then this class should not need to
     *  be touched. Using is should be fairly straightforward, though:
     *
     *  Usage:
     *  The intent of this class is to provide a unified interface and implementation
     *  to objects that expose some sort of hyper-parameters which we would like to
     *  read from a config file or the command line. In that case, make you class
     *  (called `Object` here) derive from `HyperParameterBase`.
     *  This will provide the functions `set_hyper_parameter` and `get_hyper_parameter`
     *  to query and update hyper-parameter values based on a string name.
     *
     *  How do you define which hyper-parameters exist? This is what the protected
     *  `declare_hyper_parameter` functions are for. These should be called in the
     *  constructor of your class, and declare a name for each hyper-parameter you want
     *  to use. The `HyperParameterBase` object does not manage the storage of the
     *  hyper-parameter values, just access to it. This is deliberate, as looking up
     *  a hyper-parameter by name might be a costly operation, so the actual `Object`,
     *  which anyways knows exactly which hyper-parameters it has, should not do.
     *  Instead, the hyper-parameters are stored as part of the `Object` class.
     *  For example, consider
     *  ```
     *  class Object: public HyperParameterBase {
     *      double A;
     *      int B;
     *      int get_b() const { return B; }
     *      void set_b(int value) {  check_value(value); B = value };
     *      // ...
     *  }
     *  ```
     *  In this case, `A` is a parameter which presumably can take on any valid double
     *  value, whereas the values of `B` should fulfill the `check_value` function.
     *  The corresponding constructor could look like this:
     *  ```
     *  Object::Object() {
     *      declare_hyper_parameter("A", Object::A);
     *      declare_hyper_parameter("B", &Object::get_b, &Object::set_b);
     *  }
     *  ```
     *
     *  Implementation Details:
     *  Internally, the hyper-parameters are stored as a map that maps
     *  hyper-parameter names to getter and setter functions. These functions
     *  take as first argument a pointer to the object, the setter as second
     *  argument the target value. The reason why we give the getters/setters
     *  a pointer to the object, instead of having them modify the values directly,
     *  is that the latter would break after copy/move operations on HyperParameterBase,
     *  whereas in this way we do not need to disable copy and move.
     *
     *  Even though the class itself provides no virtual functions, its destructor is
     *  declared virtual. This enables RTTI, which we use to check consistency of the
     *  supplied this pointers.
     *
     *  The magic happens in the `declare_hyper_parameter` functions. These construct the
     *  function objects, based on member pointers or pointers-to-member-function. See
     *  their documentation for details.
     *
     */
    class HyperParameterBase {
    public:
        using hyper_param_t = std::variant<long, double>;

        HyperParameterBase() = default;
        virtual ~HyperParameterBase() = default;

        /// updates the value of the hyper-parameter with the given `name`. If the type
        /// does not match the internal hyper-parameter type, an exception is throw.
        void set_hyper_parameter(const std::string& name, long value);
        void set_hyper_parameter(const std::string& name, double value);

        /// Gets the value of the named hyper-parameter. Since we cannot know the
        /// exact type, we return a variant.
        [[nodiscard]] hyper_param_t get_hyper_parameter(const std::string& name) const;

        /// Returns a vector that lists all hyper parameter names
        std::vector<std::string> get_hyper_parameter_names() const;

    protected:
        /*!
         * Declares an unconstrained hyper-parameter. The getter and setter functions are created
         * automatically and read from/write to the given member variable.
         * \tparam U The type of the hyper-parameter. Has to be one of long, double
         * \tparam S The class into which the pointer goes. This has to be the actual type (or a base therof)
         * of the this pointer, i.e. it is required that dynamic_cast<S*>(this) succeeds.
         * \param name Name of the hyper-parameter.
         * \param pointer Pointer to member of the class which stores the value of the hyper-parameter.
         */
        template<class U, class S>
        void declare_hyper_parameter(std::string name, U S::* pointer)
        {
            /*! statically, we can only test whether whether the class `S` is derived from `HyperParameterBase`, but
             * not if it is actually consistent with the `this` pointer. However, this static assert should catch almost
             * all erroneous uses. The rest will be caught at runtime in the dynamic_cast check
            */
            static_assert(std::is_base_of_v<HyperParameterBase, S>, "HyperParameterBase is not base class of member pointer");

            if(dynamic_cast<S*>(this) == nullptr) {
                throw std::logic_error("Cannot cast this pointer to class `S`");
            }

            /// OK, we know we are consistent, generate the getter and setter functions. I think technically we could
            /// get away with using static_cast here, since we know that `self` is actually of type `S`, but I think
            /// having the explicit dynamic_cast helps to emphasize what is happening.
            auto getter = [pointer](const HyperParameterBase* self) {
                return dynamic_cast<const S*>(self)->*pointer;
            };
            auto setter = [pointer](HyperParameterBase* self, const U& value) {
                dynamic_cast<S*>(self)->*pointer = value;
            };

            /// and insert them into the map
            declare_hyper_parameter(std::move(name), HyperParamData<U>{setter, getter});
        }
        /*!
         * \brief Declares an constrained hyper-parameter with explicit getter and setter function.
         * \tparam U The type of the hyper-parameter. Has to be one of long, double.
         * \tparam S The class into which the member function pointer goes. This has to be the actual type (or a base therof)
         * of the this pointer, i.e. it is required that dynamic_cast<S*>(this) succeeds.
         * \param name Name of the hyper-parameter.
         * \param getter Pointer to member of the class which reads the value of the hyper-parameter.
         * \param setter Pointer to member of the class which sets the value of the hyper-parameter.
         */
        template<class U, class S>
        void declare_hyper_parameter(std::string name, U(S::*getter)() const, void(S::*setter)(U))
        {
            static_assert(std::is_base_of_v<HyperParameterBase, S>, "T is not base class of member pointer");

            if(dynamic_cast<S*>(this) == nullptr) {
                throw std::logic_error("Cannot cast this pointer to class `S`");
            }

            auto getter_ = [getter](const HyperParameterBase* self) -> std::decay_t<U> {
                return (dynamic_cast<const S*>(self)->*getter)();
            };
            auto setter_ = [setter](HyperParameterBase* self, const U& value) {
                (dynamic_cast<S*>(self)->*setter)(value);
            };

            declare_hyper_parameter(std::move(name), HyperParamData<U>{setter_, getter_});
        }

        /*!
         * \brief Declares a sub-object that also contains hyper-parameters.
         * \details Any hyper-parameter in the sub-object will be also added to this object, where the
         * name is prefixed by the \p name parameter given to this function. At the point of this function call, the
         * sub-object must exist and have its hyper-paramters already initialized.
         * \tparam S The class into which the member function pointer goes. This has to be the actual type (or a base therof)
         * of the this pointer, i.e. it is required that dynamic_cast<S*>(this) succeeds.
         */
        template<class T, class S>
        void declare_sub_object(const std::string& name, T S::*object) {
            static_assert(std::is_base_of_v<HyperParameterBase, S>, "S is not base class of member pointer");
            static_assert(std::is_base_of_v<HyperParameterBase, T>, "T is not base class of member pointer");

            if(dynamic_cast<S*>(this) == nullptr) {
                throw std::logic_error("Cannot cast this pointer to class `S`");
            }

            // we need to get one instance of the sub-object, so we can iterate its hyper-parameters
            HyperParameterBase& sub_object = dynamic_cast<S*>(this)->*object;

            for(const auto& hp : sub_object.m_HyperParameters)
            {
                std::visit([&](const auto& hp_data)
                {
                    // get the type that is used in the inner object
                    using value_t = typename std::decay_t<decltype(hp_data)>::ValueType;
                    // make getter and setter functions that look up the actual inner objects in the
                    // submitted `self`, and then call the inner getters and setters
                    auto getter_ = [object, inner_get=hp_data.Getter](const HyperParameterBase* self) {
                        auto self_as_s = dynamic_cast<const S*>(self);
                        return inner_get(&(self_as_s->*object));
                    };
                    auto setter_ = [object, inner_set=hp_data.Setter](HyperParameterBase* self, const value_t& value) {
                        auto self_as_s = dynamic_cast<S*>(self);
                        inner_set(&(self_as_s->*object), value);
                    };
                    // register with . separated name
                    declare_hyper_parameter(name + "." + hp.first, HyperParamData<value_t>{setter_, getter_});
                    }, hp.second);
            }
         }

    private:

        /// This structure collects the Getter and Setter functions. This is what we store in the variant.
        template<class D>
        struct HyperParamData {
            std::function<void(HyperParameterBase*, D)> Setter;
            std::function<D(const HyperParameterBase*)> Getter;
            using ValueType = D;
        };

        /*! The internal `declare_hyper_parameter` function, adds the HP data to the map and checks that
         * the names are unique.
         */
        template<class D>
        void declare_hyper_parameter(std::string name, HyperParamData<D> data) {
            auto result = m_HyperParameters.insert( std::make_pair(std::move(name), std::move(data)) );
            if(!result.second) {
                throw std::logic_error("Trying to re-register hyper-parameter " + result.first->first);
            }
        }

        using hyper_param_ptr_t = std::variant<HyperParamData<long>, HyperParamData<double>>;
        std::unordered_map<std::string, hyper_param_ptr_t> m_HyperParameters;
    };


    /*!
     * \brief This class represents a set of hyper-parameters.
     */
    class HyperParameters {
    public:
        using hyper_param_t = HyperParameterBase::hyper_param_t;

        /// Sets a hyper-parameter with the given name and value
        void set(const std::string& name, long value);
        void set(const std::string& name, double value);

        /// Gets the hyper-parameter with the given name, or throws if it does not exist.
        [[nodiscard]] hyper_param_t get(const std::string& name) const;

        /// Applies the hyper-parameter values to `target`. It is valid to call this if not all hyper-parameters
        /// in target are part of this hyper-parameter set, but an error if additional parameters are present.
        void apply(HyperParameterBase& target) const;

    private:
        std::unordered_map<std::string, hyper_param_t> m_Values;
    };
}

#endif //DISMEC_HYPERPARAMS_H
