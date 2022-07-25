// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_OBJECTIVE_H
#define DISMEC_OBJECTIVE_H

#include <cstdint>
#include <memory>
#include "matrix_types.h"
#include "stats/tracked.h"
#include "fwd.h"

namespace dismec::objective
{
    /*!
     * \brief Class that models an optimization objective.
     * \details Currently, we expect objectives to be continuous with Lipschitz-continuous derivative. The following
     * operations need to be implemented by an objective:
     *  - \ref num_variables(): Gets the number of variables this objective expects for its `location` parameters.
     *  - \ref value(): Evaluates the objective at the given location.
     *  - \ref gradient(): Calculates the gradient at the given location.
     *  - \ref hessian_times_direction(): Calculates the product of the Hessian at the given location with the supplied
     *    direction vector.
     *
     * Additionally, the functions `diag_preconditioner()` and `gradient_at_zero()` do have default
     * implementations, but are expected to be implementable more usefully (for the preconditioner) or more efficiently
     * (for the gradient at zero) in the derived classes.
     *
     * These methods take the `location` parameter using a `HashVector` type. This is so that it is possible to cache
     * certain computations inside the objective, i.e. if multiple `hessian_times_direction()` calculations are
     * required for the same `location` but different `direction`. We do not use the same `Objective` instance from
     * multiple threads, so the internal caching does not need to take any synchronization into account. This is also
     * indicated by the fact that the calculation function, which do not change the visible state of the objective, are
     * not marked as const. Note that, though it
     * may be unlikely that we ask for the `value()` (or `gradient` or `hessian_times_direction()`) of the same location
     * twice (then the caller should do the caching), caching can be used to extract computations that are common
     * between `value()`, `gradient()`, `hessian_times_direction()`, and `project_to_line()`.
     */
    class Objective : public stats::Tracked {
    public:
        Objective();
        virtual ~Objective() noexcept = default;

        /*!
         * \brief Evaluate the objective at the given `location`.
         * \details The location is passed as a `HashVector` to allow caching some calculations (see detail
         * description of `Objective`).
         * \param location The location (=value of the variables) at which the objective should be evaluated.
         * This vector needs to have `location->size() == num_variables()`.
         * \return The objective's value.
         */
        [[nodiscard]] real_t value(const HashVector& location);

        //! Gets the number of variables this objective expects. May return -1 if the objective is agnostic to the
        //! number of variables, e.g. for regularizers.
        [[nodiscard]] virtual long num_variables() const noexcept = 0;

        /*!
         * \brief Get precondition to be used in CG optimization.
         * \details Calculates the diagonal of a preconditioning matrix for the Hessian and
         * places it in `target`. The default implementation returns a unit vector, i.e. does
         * not result in any preconditioning.
         * \param location The location where the corresponding hessian would be calculated.
         * \param target Pre-allocated vector where the diagonal entries of the Hessian will be placed.
         */
        void diag_preconditioner(const HashVector& location, Eigen::Ref<DenseRealVector> target);

        /*!
         * \brief creates a function g such that `g(a) = objective(location + a * direction)` Use `lookup_on_line()` to
         * evaluate `g`.
         * \details This prepares the evaluation of the objective along a line.
         * The purpose is that in many cases, this allows for much faster computations, since we can
         * cache certain results like the product of features and direction vector.
         */
        void project_to_line(const HashVector& location, const DenseRealVector& direction);

        /*!
         * \brief Looks up the value of the objective on the line defined by the last call to `project_to_line()`.
         * \param position The location where the objective is calculated.
         * \return The value of `objective(location + position * direction)`, where `location` and `direction` are the
         * vectors passed to the last call of `project_to_line()`.
         * \attention This function may use results cached `project_to_line()`, so it has to be called after a call
         * to that function. A new call to `project_to_line()` will change the line which is evaluated.
         */
        [[nodiscard]] virtual real_t lookup_on_line(real_t position) = 0;

        /*!
         * \brief State that the given vector corresponds to a certain position on the line of the last line search.
         * \details This function is a pure optimization hint. It is used in the following scenario: If several
         * computations need the product of weight vector `w` and feature matrix `X`, then we can compute this product
         * only once and use a cached value for all later invocations. This can be done by comparing the vector hashes.
         * However,  as soon as a vector is modified, these hashes are invalidated. To do an efficient line search over
         * `w' = w + t d`, we  also cache the value of `X d`, so that `X w' = X w + t X d`.
         * This function then declares that the vector given in `location` corresponds to `w + t d`, where `w` and `d`
         * are the arguments passed to the last call of `project_to_line()`.
         * \todo improve this interface, together with project_to_line, to be less error prone!
         */
         virtual void declare_vector_on_last_line([[maybe_unused]] const HashVector& location, [[maybe_unused]] real_t t) {};


        /*!
         * \brief Gets the gradient for location zero.
         * \details This operation can sometimes be implemented to be much faster than the actual gradient. Therefore,
         * we provide the option that `Objective` classes overwrite this function. Note that, if this function is
         * not implemented in a derived class, the default implementation may be rather slow, since it dynamically
         * creates the corresponding zero vector.
         */
         void gradient_at_zero(Eigen::Ref<DenseRealVector> target);

        /*!
         * \brief Evaluate the gradient at `location`.
         * \details The result will be placed in target. The location is passed as a `HashVector` to allow caching
         * some calculations (See detail description of `Objective`).
         * \param location The location (=value of the variables) at which the gradient should be computed.
         * This vector needs to have `location->size() == num_variables()`, unless `num_variables() == -1`.
         * \param target Reference to a vector in which the result will be placed. Needs to have
         * `target.size() == location->size()`.
         */
        void gradient(const HashVector& location, Eigen::Ref<DenseRealVector> target);

        /*!
         * \brief Calculates the product of the Hessian matrix at `location` with `direction`.
         * \details The result will be placed in target. The location is passed as a `HashVector` to allow caching
         * some calculations (See detail description of `Objective`). We currently don't support any caching for
         * direction, as this parameter is expected to change for each invocation.
         * \param location Where should the Hessian be calculated.
         * \param direction Vector to multiply with the Hessian.
         * \param target Reference to a buffer where the resulting vector will be placed.
         */
        void hessian_times_direction(const HashVector& location,
                                     const DenseRealVector& direction,
                                     Eigen::Ref<DenseRealVector> target);

        /*!
         * \brief Combines the calculation of gradient and pre-conditioner, which may be more efficient in some cases.
         * \details See `gradient()` and `get_diag_preconditioner()` for details. The default implementation just
         * calls these two functions.
         * \param location Value of the weights for which to get gradient and pre-conditioner.
         * \param gradient Reference to a vector in which the gradient will be placed.
         * \param pre Reference to a vector in which the pre-conditioning will be placed.
         */
        void gradient_and_pre_conditioner(const HashVector& location,
                                          Eigen::Ref<DenseRealVector> gradient,
                                          Eigen::Ref<DenseRealVector> pre);

    private:
        /// The function that does the actual value computation. This is called in `value()` after
        /// the argument has been validated.
        [[nodiscard]] virtual real_t value_unchecked(const HashVector& location) = 0;

        /// The function that does the actual gradient computation. This is called in `gradient()` after
        /// the arguments have been validated.
        virtual void gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) = 0;

        /// The function that does the actual computation. This is called in `hessian_times_direction()` after
        /// the arguments have been validated.
        virtual void hessian_times_direction_unchecked(
                const HashVector& location,
                const DenseRealVector& direction,
                Eigen::Ref<DenseRealVector> target) = 0;

        /// The function that does the actual computation. This is called in `gradient_at_zero()` after
        /// the argument has been validated. The default implementation is rather inefficient and creates a new
        /// temporary zero vector.
        virtual void gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target);

        /// The function that does the actual computation. This is called in `gradient_and_pre_conditioner()` after
        /// the arguments have been validated. The default implementation sucessively calls `gradient()` and
        /// `diag_preconditioner()`.
        virtual void gradient_and_pre_conditioner_unchecked(
                const HashVector& location,
                Eigen::Ref<DenseRealVector> gradient,
                Eigen::Ref<DenseRealVector> pre);

        /// The function that does the actual computation. This is called in `diag_preconditioner()` after
        /// the arguments have been validated. The default implementation returns ones.
        virtual void diag_preconditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target);

        /// The function that does the actual computation. This is called in `project_to_line()` after
        /// the arguments have been validated.
        virtual void project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) = 0;
    };

}

#endif //DISMEC_OBJECTIVE_H
