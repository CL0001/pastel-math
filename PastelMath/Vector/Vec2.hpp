#pragma once

/**
 * Vec2.hpp
 *
 * 2D Vector implementation for Pastel Math Library.
 * Inherits common operations from VecBase and adds 2D-specific functionality.
 *
 * Features:
 * - Multiple named accessors: (x,y), (u,v), (s,t) for different contexts
 * - 2D-specific cross product (returns scalar, represents z-component of 3D cross)
 * - Perpendicular vector operations
 * - Angle calculations
 *
 * Educational comments:
 * - Contains detailed explanations for 2D-specific operations
 * - See VecBase.hpp for common vector operations
 */

#include "VecBase.hpp"
#include <cmath>

namespace Pastel::Math
{
    template<Arithmetic T>
    struct Vec2 : Detail::VecBase<T, 2>
    {
        using Base = Detail::VecBase<T, 2>;

        [[nodiscard]] constexpr       T& x()       noexcept { return Base::data[0]; }
        [[nodiscard]] constexpr const T& x() const noexcept { return Base::data[0]; }

        [[nodiscard]] constexpr       T& y()       noexcept { return Base::data[1]; }
        [[nodiscard]] constexpr const T& y() const noexcept { return Base::data[1]; }

        [[nodiscard]] constexpr       T& u()       noexcept { return Base::data[0]; }
        [[nodiscard]] constexpr const T& u() const noexcept { return Base::data[0]; }

        [[nodiscard]] constexpr       T& v()       noexcept { return Base::data[1]; }
        [[nodiscard]] constexpr const T& v() const noexcept { return Base::data[1]; }

        [[nodiscard]] constexpr       T& s()       noexcept { return Base::data[0]; }
        [[nodiscard]] constexpr const T& s() const noexcept { return Base::data[0]; }

        [[nodiscard]] constexpr       T& t()       noexcept { return Base::data[1]; }
        [[nodiscard]] constexpr const T& t() const noexcept { return Base::data[1]; }

        constexpr Vec2() noexcept : Base() {}

        constexpr Vec2(T _x, T _y) noexcept : Base(_x, _y) {}

        constexpr Vec2(const Base& base) noexcept : Base(base) {}

        /**
         * 2D Cross Product (Perpendicular Dot Product)
         *
         * Formula: a * b = a.x * b.y - a.y * b.x
         *
         * Returns a scalar representing the z-component of the 3D cross product
         * if these 2D vectors were extended to 3D with z = 0.
         *
         * Geometric meaning:
         *  - Positive: 'b' is counter-clockwise from 'a' (left turn)
         *  - Zero:     vectors are parallel              (collinear)
         *  - Negative: 'b' is clockwise from 'a'         (right turn)
         *
         * The magnitude |a * b| equals the area of the parallelogram formed by 'a' and 'b'.
         * Also equals ||a|| * ||b|| * sin(theta) where 'theta' is the angle between vectors.
         */
        [[nodiscard]] constexpr T Cross(const Vec2& rhs) const noexcept
        {
            return x() * rhs.y() - y() * rhs.x();
        }

        /**
         * Perpendicular Vector (90° Counter-Clockwise Rotation)
         *
         * Formula: perp(x, y) = (-y, x)
         *
         * Returns a vector perpendicular to this one, rotated 90deg counter-clockwise.
         * The resulting vector has the same length as the original.
         *
         * Geometric meaning: If this vector points right, Perp() points up.
         * If this vector points up, Perp() points left, etc.
         *
         * Properties:
         * - v.Dot(v.Perp()) = 0     (always perpendicular)
         * - ||v.Perp()||    = ||v|| (same length)
         * - v.Perp().Perp() = -v    (rotating 180°)
         */
        [[nodiscard]] constexpr Vec2 Perp() const noexcept
        {
            return Vec2(-y(), x());
        }

        /**
         * Perpendicular Vector (90° Clockwise Rotation)
         *
         * Formula: perpCW(x, y) = (y, -x)
         *
         * Returns a vector perpendicular to this one, rotated 90° clockwise.
         * Opposite rotation direction from Perp().
         */
        [[nodiscard]] constexpr Vec2 PerpCW() const noexcept
        {
            return Vec2(y(), -x());
        }

        /**
         * Angle of Vector
         *
         * Formula: angle = atan2(y, x)
         *
         * Returns the angle of this vector in radians, measured from the positive x-axis.
         * Range: [-PI, PI] or [-180deg, 180deg]
         *
         * Angle convention:
         * - 0:         pointing right (1, 0)
         * - PI/2:      pointing up    (0, 1)
         * - PI or -PI: pointing left  (-1, 0)
         * - -PI/2:     pointing down  (0, -1)
         *
         * Note: atan2 handles all quadrants correctly, unlike atan.
         * atan2(0, 0) is undefined (typically returns 0).
         */
        [[nodiscard]] T Angle() const noexcept
        {
            return std::atan2(y(), x());
        }

        /**
         * Angle Between Two Vectors
         *
         * Formula: angle = atan2(cross, dot)
         * where cross = a.x * b.y - a.y * b.x
         * and dot = a.x * b.x + a.y * b.y
         *
         * Returns the signed angle from this vector to 'other' in radians.
         * Range: [-PI, PI]
         *
         * Sign convention:
         * - Positive: other is counter-clockwise from this
         * - Zero:     vectors point in same direction
         * - Negative: other is clockwise from this
         *
         * This is more robust than using just dot product because:
         * - dot only gives unsigned angle (loses direction information)
         * - atan2(cross, dot) gives both magnitude and sign
         * - Handles all edge cases (parallel, opposite, perpendicular)
         */
        [[nodiscard]] T AngleTo(const Vec2& other) const noexcept
        {
            return std::atan2(Cross(other), Base::Dot(other));
        }

        /**
         * Rotate Vector by Angle
         *
         * Formula (2D rotation matrix):
         *   x' = x * cos(theta) - y * sin(theta)
         *   y' = x * sin(theta) + y * cos(theta)
         *
         * Rotates this vector counter-clockwise by the given angle in radians.
         *
         * Angle convention:
         * - Positive angle: counter-clockwise rotation
         * - Negative angle: clockwise rotation
         * - 0:              no rotation
         *
         * The magnitude (length) is preserved during rotation.
         *
         * Performance note: sin/cos are relatively expensive.
         * If rotating many vectors by the same angle, compute sin/cos once
         * and use RotateFast().
         */
        [[nodiscard]] Vec2 Rotate(T angle_radians) const noexcept
        {
            T cos_angle = std::cos(angle_radians);
            T sin_angle = std::sin(angle_radians);

            return Vec2(
                x() * cos_angle - y() * sin_angle,
                x() * sin_angle + y() * cos_angle
            );
        }

        /**
         * Rotate Vector by Precomputed Sin/Cos
         *
         * Same as Rotate() but with precomputed sine and cosine values.
         * Use this when rotating many vectors by the same angle for better performance.
         */
        [[nodiscard]] constexpr Vec2 RotateFast(T cos_angle, T sin_angle) const noexcept
        {
            return Vec2(
                x() * cos_angle - y() * sin_angle,
                x() * sin_angle + y() * cos_angle
            );
        }

        /**
         * Create Vector from Angle
         *
         * Formula: (cos(θ), sin(θ))
         *
         * Static factory method that creates a unit vector pointing in the
         * direction specified by the angle in radians.
         *
         * Angle convention:
         * - 0: points right (1, 0)
         * - π/2: points up (0, 1)
         * - π: points left (-1, 0)
         * - 3π/2: points down (0, -1)
         *
         * The resulting vector always has magnitude = 1 (unit vector).
         */
        [[nodiscard]] static Vec2 FromAngle(T angle_radians) noexcept
        {
            return Vec2(std::cos(angle_radians), std::sin(angle_radians));
        }

        /**
         * Signed Distance to Line
         *
         * Returns the signed perpendicular distance from this point to a line
         * defined by two points (lineStart and lineEnd).
         *
         * Sign convention:
         * - Positive: point is on the left side of the line (counter-clockwise)
         * - Zero: point is exactly on the line
         * - Negative: point is on the right side of the line (clockwise)
         *
         * Formula: Uses cross product to determine signed area, then divides
         * by line length to get perpendicular distance.
         *
         * "Left/right" is determined by the direction from lineStart to lineEnd.
         */
        [[nodiscard]] T SignedDistanceToLine(const Vec2& line_start, const Vec2& line_end) const noexcept
        {
            Vec2 line_dir = line_end - line_start;
            Vec2 point_dir = *this - line_start;
            return point_dir.Cross(line_dir) / line_dir.Magnitude();
        }
    };

    using Vec2f = Vec2<float>;
    using Vec2d = Vec2<double>;
    using Vec2i = Vec2<int>;
    using Vec2u = Vec2<unsigned int>;

    template<Arithmetic T>
    [[nodiscard]] constexpr T Cross(const Vec2<T>& a, const Vec2<T>& b) noexcept
    {
        return a.Cross(b);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr T Dot(const Vec2<T>& a, const Vec2<T>& b) noexcept
    {
        return a.Dot(b);
    }

    template<Arithmetic T>
    [[nodiscard]] Vec2<T> Normalize(const Vec2<T>& v) noexcept
    {
        return v.Normalize();
    }

    template<Arithmetic T>
    [[nodiscard]] T Distance(const Vec2<T>& a, const Vec2<T>& b) noexcept
    {
        return a.Distance(b);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr Vec2<T> Lerp(const Vec2<T>& a, const Vec2<T>& b, T t) noexcept
    {
        return a.Lerp(b, t);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr Vec2<T> Perp(const Vec2<T>& v) noexcept
    {
        return v.Perp();
    }

    template<Arithmetic T>
    [[nodiscard]] Vec2<T> Rotate(const Vec2<T>& v, T angleRadians) noexcept
    {
        return v.Rotate(angleRadians);
    }
} // namespace Pastel::Math