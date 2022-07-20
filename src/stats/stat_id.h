// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_STATS_STAT_ID_H
#define DISMEC_STATS_STAT_ID_H

#include <cstdint>
#include <string>
#include "utils/opaque_int.h"

namespace dismec::stats {
    namespace detail {
        /// A tag struct to define an opaque int to identify a statistics
        struct stat_id_tag;
        /// A tag struct to define an opaque int to identify a tag
        struct tag_id_tag;
    }

    /*!
     * \brief An opaque int-like type that is used to identify a statistic in a `StatisticsCollection`.
     */
    using stat_id_t = opaque_int_type<detail::stat_id_tag>;
    /*!
     * \brief An opaque int-like type that is used to identify a tag in a `StatisticsCollection`.
     */
    using tag_id_t = opaque_int_type<detail::tag_id_tag>;

    /*!
     * \brief Data that is associated with each declared statistics.
     */
    struct StatisticMetaData {
        std::string Name;   //!< The name of the stat. This is how it will be identified.
        std::string Unit;   //!< The unit in which the data points will be supplied. What you would put on the x-axis when plotting a histogram.
    };
}

#endif //DISMEC_STATS_STAT_ID_H
