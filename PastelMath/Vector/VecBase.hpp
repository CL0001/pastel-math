#pragma once

/**
 * VecBase.hpp
 *
 * Generic N-dimensional vector base for Pastel Math Library.
 * Provides common vector operations shared by Vec2, Vec3, and Vec4.
 *
 * This class is not intended to be used directly.
 * Dimension-specific functionality (named components, cross products, etc.)
 * is implemented in derived vector types.
 *
 * Implementation notes:
 * - Uses raw array for storage (not std::array) for union compatibility
 * - All operations are constexpr where possible
 * - Member functions are preferred, free functions delegate to members
 * - Zero vectors are returned for invalid operations (e.g., normalize zero vector)
 * 
 * Educational comments:
 * - Focuses on dimension-independent vector math
 * - Explains geometric meaning where applicable
 * - Dimension-specific explanations are deferred to derived types
 */

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <concepts>
#include <type_traits>

namespace Pastel::Math
{
    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

    namespace Detail
    {
        template<Arithmetic T, std::size_t N>
        struct VecBase
        {
            T data[N];

            constexpr VecBase() noexcept : data{} {}

            template<typename... Args>
            requires (sizeof...(Args) == N && (std::convertible_to<Args, T> && ...))
            constexpr VecBase(Args... args) noexcept : data{ static_cast<T>(args)... } {}

            [[nodiscard]] constexpr T& operator[](std::size_t i) noexcept
            {
                return data[i];
            }

            [[nodiscard]] constexpr const T& operator[](std::size_t i) const noexcept
            {
                return data[i];
            }

            /**
             * Magnitude Squared
             *
             * Formula: ||v||^2 = x^2 + y^2 + z^2 + ...
             *
             * Returns the squared length of the vector without computing the square root.
             * This is faster than Magnitude() and should be preferred when:
             * - Comparing distances
             * - Checking if length is within threshold
             * - The actual length value isn't needed
             */
            [[nodiscard]] constexpr T MagnitudeSq() const noexcept
            {
                T sum = T(0);

                for (std::size_t i = 0; i < N; ++i)
                {
                    sum += data[i] * data[i];
                }

                return sum;
            }

            /**
             * Magnitude
             *
             * Formula: ||v|| = sqrt(x^2 + y^2 + z^2 + ...)
             *
             * Returns the Euclidean length of the vector. This is the straight-line
             * distance from the origin to the point represented by this vector.
             *
             * Geometric meaning: The magnitude represents how long the vector is.
             * A unit vector has magnitude equal to 1, and the zero vector has magnitude of 0.
             *
             * Performance note: Uses sqrt which is relatively expensive. Consider
             * using MagnitudeSq() if you only need to compare lengths.
             */
            [[nodiscard]] T Magnitude() const noexcept
            {
                return std::sqrt(MagnitudeSq());
            }

            /**
             * Distance Squared
             *
             * Formula: dist^2 = (x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2 + ...
             *
             * Returns the squared Euclidean distance between two points (this and other).
             * Equivalent to (this - other).MagnitudeSq() but computed directly.
             *
             * Use this instead of Distance() when comparing distances to avoid sqrt.
             */
            [[nodiscard]] constexpr T DistanceSq(const VecBase& other) const noexcept
            {
                T sum = T(0);

                for (std::size_t i = 0; i < N; ++i)
                {
                    T diff = data[i] - other.data[i];
                    sum += diff * diff;
                }

                return sum;
            }

            /**
             * Distance (Euclidean Distance)
             *
             * Formula: dist = sqrt((x_1 - x_1)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2 + ...)
             *
             * Returns the straight-line distance between two points in N-dimensional space.
             * This is the standard distance formula taught in geometry.
             */
            [[nodiscard]] T Distance(const VecBase& other) const noexcept
            {
                return std::sqrt(DistanceSq(other));
            }

            /**
             * Normalize (Unit Vector)
             *
             * Formula: unit vector = v / ||v||
             *
             * Returns a vector with the same direction but length = 1 (unit vector).
             *
             * Geometric meaning: Points in the same direction as the original vector
             * but is exactly 1 unit long. Think of it as the "direction" of the vector.
             *
             * Special case: If the vector has zero length,
             * returns the zero vector to avoid division by zero.
             */
            [[nodiscard]] VecBase Normalize() const noexcept
            {
                T mag = Magnitude();

                if (mag == T(0))
                {
                    return VecBase{};
                }

                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = data[i] / mag;
                }

                return result;
            }

            /**
             * Safe Normalize
             *
             * Like Normalize(), but returns a specified fallback vector instead of
             * the zero vector when magnitude is zero.
             *
             * Useful when you need a guaranteed valid direction even for zero vectors.
             * Example: Getting a default "up" direction if calculated normal is degenerate.
             */
            [[nodiscard]] VecBase SafeNormalize(const VecBase& fallback = VecBase{}) const noexcept
            {
                T mag = Magnitude();

                if (mag == T(0))
                {
                    return fallback;
                }

                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = data[i] / mag;
                }

                return result;
            }

            /**
             * Dot Product
             *
             * Formula: a dot b = a_1 * b_1 + a_2 * b_2 + a_3 * b_3 + ...
             *
             * Geometric formula: a dot b = ||a|| * ||b|| cos(theta)
             * where 'theta' is the angle between the vectors
             *
             * Returns a scalar representing how much the vectors align:
             * - Positive (angle < 90�): vectors point in generally the same direction
             * - Zero     (angle = 90�): vectors are perpendicular
             * - Negative (angle > 90�): vectors point in opposite directions
             *
             * Special cases:
             * - a dot a = ||a||^2 (dot product with self equals magnitude squared)
             * - If both vectors are normalized: a dot b = cos(theta)
             */
            [[nodiscard]] constexpr T Dot(const VecBase& other) const noexcept
            {
                T sum = T(0);

                for (std::size_t i = 0; i < N; ++i)
                {
                    sum += data[i] * other.data[i];
                }

                return sum;
            }

            /**
             * Vector Projection
             *
             * Formula: proj_b(a) = ((a dot b) / (b dot b)) * b
             *
             * Projects this vector onto the 'onto' vector, returning the component
             * of 'this' that lies along the direction of 'onto'.
             *
             * Geometric meaning: If you shine a light perpendicular to vector 'onto',
             * the projection is the shadow of 'this' vector cast onto 'onto'.
             *
             * The result is a vector that:
             * - Points in the same direction as 'onto' (or opposite if projection is negative)
             * - Has length = ||this|| * cos(theta), where 'theta' is angle between vectors
             *
             * Special case: Returns zero vector if projecting onto a zero vector
             * (division by zero is avoided).
             */
            [[nodiscard]] constexpr VecBase Project(const VecBase& onto) const noexcept
            {
                T denominator = onto.Dot(onto);

                if (denominator == T(0))
                {
                    return VecBase{};
                }

                T scalar = Dot(onto) / denominator;
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = onto.data[i] * scalar;
                }

                return result;
            }

            /**
             * Vector Rejection
             *
             * Formula: reject_b(a) = a - proj_b(a)
             *
             * Returns the component of this vector that is perpendicular to 'from'.
             * This is the "leftover" part after removing the projection.
             *
             * Geometric meaning: The part of 'this' that doesn't lie along 'from'.
             *
             * Properties:
             * - Project(v) + Reject(v) = this (they are complementary)
             * - Reject(v) is perpendicular to v: Reject(v) dot v = 0
             */
            [[nodiscard]] constexpr VecBase Reject(const VecBase& from) const noexcept
            {
                return *this - Project(from);
            }

            /**
             * Vector Reflection
             *
             * Formula: r = v - 2 * (v dot n) * n
             * where 'v' is this vector, 'n' is the normal, 'r' is the reflection
             *
             * Reflects this vector across a surface defined by its normal vector.
             * The normal should be normalized for correct results.
             *
             * Geometric meaning: Like a ball bouncing off a wall, the angle of
             * incidence equals the angle of reflection. The component perpendicular
             * to the surface reverses, while the parallel component stays the same.
             *
             * Properties:
             * - ||reflected|| = ||original|| (magnitude is preserved)
             * - If incoming, normal point toward each other: result points away
             * - Angle in = angle out (relative to normal)
             */
            [[nodiscard]] constexpr VecBase Reflect(const VecBase& normal) const noexcept
            {
                T factor = T(2) * Dot(normal);
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = data[i] - factor * normal.data[i];
                }

                return result;
            }

            /**
             * Vector Refraction (Snell's Law)
             *
             * Formula: Based on Snell's Law: eta_1 * sin(theta_1) = eta_2 * sin(theta_2)
             * where 'eta' is the refractive index, 'theta' is angle from normal
             *
             * Calculates the refracted direction when a ray passes through the
             * boundary between two media with different refractive indices.
             *
             * Parameters:
             * - normal: Surface normal at intersection point
             * - eta: Ratio of refractive indices (eta_1 / eta_2)
             *   - Air to water: ~0.75 (1.0/1.33)
             *   - Water to air: ~1.33 (1.33/1.0)
             *   - Air to glass: ~0.67 (1.0/1.5)
             *   - Glass to air: ~1.5  (1.5/1.0)
             *
             * Physics: Light bends when entering a different medium because it
             * travels at different speeds in different materials. The refractive
             * index describes how much the material slows down light.
             *
             * Total Internal Reflection:
             * When light tries to exit a denser medium at a steep angle,
             * it cannot escape and reflects instead. This function returns a zero
             * vector when this occurs (when sin^2(theta_2) > 1).
             */
            [[nodiscard]] VecBase Refract(const VecBase& normal, T eta) const noexcept
            {
                T cos_i = -Dot(normal);
                T sin_t2 = eta * eta * (T(1) - cos_i * cos_i);

                if (sin_t2 > T(1))
                {
                    return VecBase{};
                }

                T cos_t = std::sqrt(T(1) - sin_t2);
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = eta * data[i] + (eta * cos_i - cos_t) * normal.data[i];
                }

                return result;
            }

            /**
             * Linear Interpolation (Lerp)
             *
             * Formula: result = (1 - t) * this + t * other
             * Equivalent to: result = this + t * (other - this)
             *
             * Smoothly interpolates between 'this' vector and 'other' based on parameter 't'.
             * Creates a straight-line path between the two vectors.
             *
             * Parameter 't' behavior:
             * - t = 0.0 -> returns 'this' vector (start point)
             * - t = 0.5 -> returns midpoint between vectors
             * - t = 1.0 -> returns 'other' vector (end point)
             * - t outside [0, 1] extrapolates beyond the endpoints
             *
             * Geometric meaning: Finds a point that is t% of the way from 'this'
             * to 'other' along a straight line.
             */
            [[nodiscard]] constexpr VecBase Lerp(const VecBase& other, T t) const noexcept
            {
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = data[i] + t * (other.data[i] - data[i]);
                }

                return result;
            }

            [[nodiscard]] constexpr bool operator==(const VecBase& other) const noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    if (data[i] != other.data[i])
                    {
                        return false;
                    }
                }

                return true;
            }

            [[nodiscard]] constexpr bool operator!=(const VecBase& other) const noexcept
            {
                return !(*this == other);
            }

            [[nodiscard]] bool ApproxEqual(const VecBase& other, T epsilon = T(1e-6)) const noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    T diff = std::abs(data[i] - other.data[i]);

                    if (diff > epsilon)
                    {
                        return false;
                    }
                }

                return true;
            }

            [[nodiscard]] constexpr VecBase operator-() const noexcept
            {
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = -data[i];
                }

                return result;
            }

            [[nodiscard]] constexpr VecBase operator+() const noexcept
            {
                return *this;
            }

            constexpr VecBase& operator+=(const VecBase& rhs) noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    data[i] += rhs.data[i];
                }

                return *this;
            }

            constexpr VecBase& operator-=(const VecBase& rhs) noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    data[i] -= rhs.data[i];
                }

                return *this;
            }

            constexpr VecBase& operator*=(const VecBase& rhs) noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    data[i] *= rhs.data[i];
                }
                
                return *this;
            }

            constexpr VecBase& operator/=(const VecBase& rhs) noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    data[i] /= rhs.data[i];
                }

                return *this;
            }

            constexpr VecBase& operator+=(T scalar) noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    data[i] += scalar;
                }

                return *this;
            }

            constexpr VecBase& operator-=(T scalar) noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    data[i] -= scalar;
                }

                return *this;
            }

            constexpr VecBase& operator*=(T scalar) noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    data[i] *= scalar;
                }

                return *this;
            }

            constexpr VecBase& operator/=(T scalar) noexcept
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    data[i] /= scalar;
                }

                return *this;
            }

            [[nodiscard]] constexpr VecBase Abs() const noexcept
            {
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = std::abs(data[i]);
                }

                return result;
            }

            [[nodiscard]] constexpr VecBase Min(const VecBase& other) const noexcept
            {
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = std::min(data[i], other.data[i]);
                }

                return result;
            }

            [[nodiscard]] constexpr VecBase Max(const VecBase& other) const noexcept
            {
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = std::max(data[i], other.data[i]);
                }

                return result;
            }

            [[nodiscard]] constexpr VecBase Clamp(const VecBase& min, const VecBase& max) const noexcept
            {
                VecBase result;

                for (std::size_t i = 0; i < N; ++i)
                {
                    result.data[i] = std::clamp(data[i], min.data[i], max.data[i]);
                }

                return result;
            }

            [[nodiscard]] constexpr T MinComponent() const noexcept
            {
                T min_val = data[0];

                for (std::size_t i = 1; i < N; ++i)
                {
                    min_val = std::min(min_val, data[i]);
                }

                return min_val;
            }

            [[nodiscard]] constexpr T MaxComponent() const noexcept
            {
                T max_val = data[0];

                for (std::size_t i = 1; i < N; ++i)
                {
                    max_val = std::max(max_val, data[i]);
                }
                
                return max_val;
            }
        };

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator+(VecBase<T, N> lhs, const VecBase<T, N>& rhs) noexcept
        {
            return lhs += rhs;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator-(VecBase<T, N> lhs, const VecBase<T, N>& rhs) noexcept
        {
            return lhs -= rhs;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator*(VecBase<T, N> lhs, const VecBase<T, N>& rhs) noexcept
        {
            return lhs *= rhs;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator/(VecBase<T, N> lhs, const VecBase<T, N>& rhs) noexcept
        {
            return lhs /= rhs;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator+(VecBase<T, N> vec, T scalar) noexcept
        {
            return vec += scalar;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator+(T scalar, VecBase<T, N> vec) noexcept
        {
            return vec += scalar;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator-(VecBase<T, N> vec, T scalar) noexcept
        {
            return vec -= scalar;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator*(VecBase<T, N> vec, T scalar) noexcept
        {
            return vec *= scalar;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator*(T scalar, VecBase<T, N> vec) noexcept
        {
            return vec *= scalar;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> operator/(VecBase<T, N> vec, T scalar) noexcept
        {
            return vec /= scalar;
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr T Dot(const VecBase<T, N>& a, const VecBase<T, N>& b) noexcept
        {
            return a.Dot(b);
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] T Distance(const VecBase<T, N>& a, const VecBase<T, N>& b) noexcept
        {
            return a.Distance(b);
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr T DistanceSq(const VecBase<T, N>& a, const VecBase<T, N>& b) noexcept
        {
            return a.DistanceSq(b);
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] VecBase<T, N> Normalize(const VecBase<T, N>& v) noexcept
        {
            return v.Normalize();
        }

        template<Arithmetic T, std::size_t N>
        [[nodiscard]] constexpr VecBase<T, N> Lerp(const VecBase<T, N>& a, const VecBase<T, N>& b, T t) noexcept
        {
            return a.Lerp(b, t);
        }
    } // namespace Pastel::Math::Detail
} // namespace Pastel::Math
