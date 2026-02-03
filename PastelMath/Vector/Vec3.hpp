#pragma once

/**
 * Vec3.hpp
 *
 * 3D Vector implementation for Pastel Math Library.
 * Inherits common operations from VecBase and adds 3D-specific functionality.
 *
 * Features:
 * - Multiple named accessors: (x,y,z), (r,g,b) for different contexts
 * - 3D cross product (returns vector, perpendicular to both input vectors)
 * - Gram-Schmidt orthonormalization
 * - Spherical coordinate conversion
 *
 * Educational comments:
 * - Contains detailed explanations for 3D-specific operations
 * - See VecBase.hpp for common vector operations (dot, normalize, etc.)
 */

#include "VecBase.hpp"
#include <cmath>

namespace Pastel::Math
{
    template<Arithmetic T>
    struct Vec3 : Detail::VecBase<T, 3>
    {
        using Base = Detail::VecBase<T, 3>;

        [[nodiscard]] constexpr       T& x()       noexcept { return Base::data[0]; }
        [[nodiscard]] constexpr const T& x() const noexcept { return Base::data[0]; }

        [[nodiscard]] constexpr       T& y()       noexcept { return Base::data[1]; }
        [[nodiscard]] constexpr const T& y() const noexcept { return Base::data[1]; }

        [[nodiscard]] constexpr       T& z()       noexcept { return Base::data[2]; }
        [[nodiscard]] constexpr const T& z() const noexcept { return Base::data[2]; }

        [[nodiscard]] constexpr       T& r()       noexcept { return Base::data[0]; }
        [[nodiscard]] constexpr const T& r() const noexcept { return Base::data[0]; }

        [[nodiscard]] constexpr       T& g()       noexcept { return Base::data[1]; }
        [[nodiscard]] constexpr const T& g() const noexcept { return Base::data[1]; }

        [[nodiscard]] constexpr       T& b()       noexcept { return Base::data[2]; }
        [[nodiscard]] constexpr const T& b() const noexcept { return Base::data[2]; }

        constexpr Vec3() noexcept : Base() {}

        constexpr Vec3(T _x, T _y, T _z) noexcept : Base(_x, _y, _z) {}

        constexpr Vec3(const Base& base) noexcept : Base(base) {}

        /**
         * 3D Cross Product (Vector Product)
         *
         * Formula: a * b = (a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
         *
         * Returns a vector perpendicular to both input vectors.
         * The result follows the right-hand rule: point fingers along 'a',
         * curl them towards 'b', thumb points in direction of a * b.
         *
         * Geometric meaning:
         *  - The resulting vector is perpendicular to the plane formed by 'a' and 'b'
         *  - Magnitude: ||a * b|| = ||a|| * ||b|| * sin(theta), where 'theta' is angle between vectors
         *  - The magnitude equals the area of the parallelogram formed by 'a' and 'b'
         *  - Direction follows right-hand rule
         *
         * Properties:
         *  - a * b = -(b * a) (anti-commutative)
         *  - a * a = 0        (cross product with self is zero)
         *  - a * b = 0        (if vectors are parallel)
         *  - Result is perpendicular to both if a * b is perpendicular to both 'a' and 'b'
         */
        [[nodiscard]] constexpr Vec3 Cross(const Vec3& rhs) const noexcept
        {
            return Vec3(
                y() * rhs.z() - z() * rhs.y(),
                z() * rhs.x() - x() * rhs.z(),
                x() * rhs.y() - y() * rhs.x()
            );
        }

        /**
         * Triple Scalar Product
         *
         * Formula: a dot (b * c)
         *
         * Returns the signed volume of the parallelepiped formed by three vectors.
         *
         * Geometric meaning:
         *  - Positive:  vectors form a right-handed coordinate system
         *  - Zero:      vectors are coplanar (lie in the same plane)
         *  - Negative:  vectors form a left-handed coordinate system
         *  - |result| = volume of parallelepiped with edges a, b, c
         *
         * Properties:
         *  - Cyclic permutation preserves value: a dot (b * c) = b dot (c * a) = c dot (a * b)
         *  - Swapping any two vectors negates result: a dot (b * c) = -a dot (c * b)
         */
        [[nodiscard]] constexpr T ScalarTriple(const Vec3& b, const Vec3& c) const noexcept
        {
            return Base::Dot(b.Cross(c));
        }

        /**
         * Angle Between Two Vectors (3D)
         *
         * Formula: theta = arccos((a dot b) / (||a|| * ||b||))
         *
         * Returns the unsigned angle between two vectors in radians.
         * Range: [0, PI] or [0deg, 180deg]
         *
         * The angle is always positive (unsigned). If you need signed angle,
         * you must specify a reference plane/normal.
         *
         * Properties:
         *  - Returns 0      if vectors point in same direction
         *  - Returns PI     if vectors point in opposite directions
         *  - Returns PI / 2 if vectors are perpendicular
         *
         * Note: For normalized vectors, this simplifies to arccos(a dot b)
         */
        [[nodiscard]] T AngleTo(const Vec3& other) const noexcept
        {
            T dot = Base::Dot(other);
            T magProduct = Base::Magnitude() * other.Magnitude();

            if (magProduct == T(0))
            {
                return T(0);
            }

            T cosAngle = std::clamp(dot / magProduct, T(-1), T(1));

            return std::acos(cosAngle);
        }

        /**
         * Signed Angle Around Axis
         *
         * Returns the signed angle from this vector to 'other' around the given axis.
         * Range: [-PI, PI]
         *
         * Parameters:
         *  - other: target vector
         *  - axis: rotation axis (should be normalized for correct results)
         *
         * Sign convention:
         *  - Positive: counter-clockwise rotation when looking down the axis
         *  - Negative: clockwise rotation when looking down the axis
         *  - Uses right-hand rule
         *
         * Formula: Uses atan2 with dot product (for magnitude) and
         * triple scalar product (for sign/direction).
         */
        [[nodiscard]] T SignedAngle(const Vec3& other, const Vec3& axis) const noexcept
        {
            T dot = Base::Dot(other);

            Vec3 cross = Cross(other);

            T det = axis.Dot(cross);

            return std::atan2(det, dot);
        }

        /**
         * Rotate Around Axis (Rodrigues' Rotation Formula)
         *
         * Formula: v_rot = v * cos(theta) + (k * v) * sin(theta) + k * (k dot v) * (1 - cos(theta))
         * where 'k' is the normalized axis, 'theta' is the angle
         *
         * Rotates this vector around an arbitrary axis by the given angle.
         *
         * Parameters:
         *  - axis: rotation axis (should be normalized)
         *  - angleRadians: rotation angle (positive = right-hand rule direction)
         *
         * The rotation follows the right-hand rule: point thumb along axis,
         * fingers curl in the direction of positive rotation.
         *
         * Properties:
         *  - Magnitude is preserved: ||rotated|| = ||original||
         *  - Components parallel to axis are unchanged
         *  - Components perpendicular to axis are rotated
         *
         * Performance note: If rotating many vectors by the same angle/axis,
         * consider using a rotation matrix or quaternion instead.
         */
        [[nodiscard]] Vec3 RotateAround(const Vec3& axis, T angle_radians) const noexcept
        {
            T cos_angle = std::cos(angle_radians);
            T sin_angle = std::sin(angle_radians);

            Vec3 k = axis.Normalize();

            return (*this) * cos_angle + k.Cross(*this) * sin_angle + k * (k.Dot(*this)) * (T(1) - cos_angle);
        }

        /**
         * Gram-Schmidt Orthogonalization
         *
         * Given a reference vector, returns a vector perpendicular to it.
         * The result lies in the plane defined by 'this' and 'reference'.
         *
         * Formula: result = this - Project(reference)
         *
         * This is the Gram-Schmidt process for creating an orthogonal vector.
         * The result is perpendicular to 'reference' while staying close to 'this'.
         *
         * Note: Result is NOT normalized. Use .Normalize() if needed.
         * Returns zero vector if 'this' is parallel to 'reference'.
         */
        [[nodiscard]] constexpr Vec3 Orthogonalize(const Vec3& reference) const noexcept
        {
            return *this - Base::Project(reference);
        }

        /**
         * Spherical Coordinates (Azimuth Longitude)
         *
         * Formula: PHI = atan2(y, x)
         *
         * Returns the azimuth angle in radians (rotation around Z-axis).
         * Range: [-PI, PI]
         *
         * Spherical coordinate system:
         *  - Azimuth   (PHI):  angle in XY plane from +X axis
         *  - Elevation (Etha): angle from XY plane (or from +Z axis, depending on convention)
         *  - Radius    (r):    distance from origin
         *
         * Convention used here:
         *  - PHI = 0:         pointing along +X axis
         *  - PHI = PI / 2:    pointing along +Y axis
         *  - PHI = PI or -PI: pointing along -X axis
         *  - PHI = -PI / 2:   pointing along -Y axis
         */
        [[nodiscard]] T Azimuth() const noexcept
        {
            return std::atan2(y(), x());
        }

        /**
         * Spherical Coordinates - Get Elevation (Latitude)
         *
         * Formula: theta = asin(z / ||v||)
         *
         * Returns the elevation angle in radians (angle from XY plane).
         * Range: [-PI / 2, PI / 2]
         *
         * Convention:
         * - θ = 0:      vector lies in XY plane (equator)
         * - θ = π / 2:  pointing straight up    (+Z)
         * - θ = -π / 2: pointing straight down  (-Z)
         *
         * Note: Requires vector to be non-zero. Returns 0 for zero vector.
         */
        [[nodiscard]] T Elevation() const noexcept
        {
            T mag = Base::Magnitude();

            if (mag == T(0))
            {
                return T(0);
            }

            return std::asin(std::clamp(z() / mag, T(-1), T(1)));
        }

        /**
         * Create Vector from Spherical Coordinates
         *
         * Formula:
         *   x = r * cos(elevation) * cos(azimuth)
         *   y = r * cos(elevation) * sin(azimuth)
         *   z = r * sin(elevation)
         *
         * Creates a vector from spherical coordinates.
         *
         * Parameters:
         *  - azimuth:   angle in XY plane from +X axis (radians)
         *  - elevation: angle from XY plane            (radians)
         *  - radius:    distance from origin           (default = 1 for unit vector)
         */
        [[nodiscard]] static Vec3 FromSpherical(T azimuth, T elevation, T radius = T(1)) noexcept
        {
            T cos_elev = std::cos(elevation);

            return Vec3(
                radius * cos_elev * std::cos(azimuth),
                radius * cos_elev * std::sin(azimuth),
                radius * std::sin(elevation)
            );
        }

        /**
         * Component-wise Absolute Value
         *
         * Returns a vector where each component is the absolute value of the
         * corresponding component in this vector.
         */
        [[nodiscard]] Vec3 Abs() const noexcept
        {
            return Vec3(std::abs(x()), std::abs(y()), std::abs(z()));
        }
    };

    using Vec3f = Vec3<float>;
    using Vec3d = Vec3<double>;
    using Vec3i = Vec3<int>;
    using Vec3u = Vec3<unsigned int>;

    template<Arithmetic T>
    [[nodiscard]] constexpr Vec3<T> Cross(const Vec3<T>& a, const Vec3<T>& b) noexcept
    {
        return a.Cross(b);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr T Dot(const Vec3<T>& a, const Vec3<T>& b) noexcept
    {
        return a.Dot(b);
    }

    template<Arithmetic T>
    [[nodiscard]] Vec3<T> Normalize(const Vec3<T>& v) noexcept
    {
        return v.Normalize();
    }

    template<Arithmetic T>
    [[nodiscard]] T Distance(const Vec3<T>& a, const Vec3<T>& b) noexcept
    {
        return a.Distance(b);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr Vec3<T> Lerp(const Vec3<T>& a, const Vec3<T>& b, T t) noexcept
    {
        return a.Lerp(b, t);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr T ScalarTriple(const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c) noexcept
    {
        return a.ScalarTriple(b, c);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr Vec3<T> Orthogonalize(const Vec3<T>& v, const Vec3<T>& reference) noexcept
    {
        return v.Orthogonalize(reference);
    }

    template<Arithmetic T>
    inline constexpr Vec3<T> Vec3Zero = Vec3<T>(T(0), T(0), T(0));

    template<Arithmetic T>
    inline constexpr Vec3<T> Vec3One = Vec3<T>(T(1), T(1), T(1));

    template<Arithmetic T>
    inline constexpr Vec3<T> Vec3Right = Vec3<T>(T(1), T(0), T(0));

    template<Arithmetic T>
    inline constexpr Vec3<T> Vec3Up = Vec3<T>(T(0), T(1), T(0));

    template<Arithmetic T>
    inline constexpr Vec3<T> Vec3Forward = Vec3<T>(T(0), T(0), T(1));

    inline constexpr Vec3f Vec3fZero    = Vec3Zero<float>;
    inline constexpr Vec3f Vec3fOne     = Vec3One<float>;
    inline constexpr Vec3f Vec3fRight   = Vec3Right<float>;
    inline constexpr Vec3f Vec3fUp      = Vec3Up<float>;
    inline constexpr Vec3f Vec3fForward = Vec3Forward<float>;
} // namespace Pastel::Math