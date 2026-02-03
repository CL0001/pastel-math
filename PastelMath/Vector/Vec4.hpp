#pragma once

/**
 * Vec4.hpp
 *
 * 4D Vector implementation for Pastel Math Library.
 * Inherits common operations from VecBase and adds 4D-specific functionality.
 *
 * Features:
 *  - Multiple named accessors: (x, y, z, w), (r, g, b, a) for different contexts
 *  - Homogeneous coordinate operations (perspective divide, conversion to/from Vec3)
 *  - Color operations when used as RGBA
 *  - Swizzling for component reordering
 *
 * Educational comments:
 *  - Contains detailed explanations for 4D-specific operations
 *  - See VecBase.hpp for common vector operations (dot, normalize, etc.)
 *  - See Vec3.hpp for 3D-specific operations
 *
 * The W component:
 *  - In 3D graphics, Vec4 is primarily used for homogeneous coordinates
 *  - w = 1: represents a point in 3D space
 *  - w = 0: represents a direction
 *  - Other w values: used for perspective projection
 */

#include "VecBase.hpp"
#include "Vec3.hpp"
#include <cmath>

namespace Pastel::Math
{
    template<Arithmetic T>
    struct Vec4 : Detail::VecBase<T, 4>
    {
        using Base = Detail::VecBase<T, 4>;

        [[nodiscard]] constexpr       T& x()       noexcept { return Base::data[0]; }
        [[nodiscard]] constexpr const T& x() const noexcept { return Base::data[0]; }

        [[nodiscard]] constexpr       T& y()       noexcept { return Base::data[1]; }
        [[nodiscard]] constexpr const T& y() const noexcept { return Base::data[1]; }

        [[nodiscard]] constexpr       T& z()       noexcept { return Base::data[2]; }
        [[nodiscard]] constexpr const T& z() const noexcept { return Base::data[2]; }

        [[nodiscard]] constexpr       T& w()       noexcept { return Base::data[3]; }
        [[nodiscard]] constexpr const T& w() const noexcept { return Base::data[3]; }

        [[nodiscard]] constexpr       T& r()       noexcept { return Base::data[0]; }
        [[nodiscard]] constexpr const T& r() const noexcept { return Base::data[0]; }

        [[nodiscard]] constexpr       T& g()       noexcept { return Base::data[1]; }
        [[nodiscard]] constexpr const T& g() const noexcept { return Base::data[1]; }

        [[nodiscard]] constexpr       T& b()       noexcept { return Base::data[2]; }
        [[nodiscard]] constexpr const T& b() const noexcept { return Base::data[2]; }

        [[nodiscard]] constexpr       T& a()       noexcept { return Base::data[3]; }
        [[nodiscard]] constexpr const T& a() const noexcept { return Base::data[3]; }

        constexpr Vec4() noexcept : Base() {}

        constexpr Vec4(T _x, T _y, T _z, T _w) noexcept : Base(_x, _y, _z, _w) {}

        constexpr Vec4(const Base& base) noexcept : Base(base) {}

        constexpr Vec4(const Vec3<T>& v, T _w = T(1)) noexcept : Base(v.x(), v.y(), v.z(), _w) {}

        /**
         * Perspective Divide (Homogeneous to Cartesian)
         *
         * Formula: result = (x / w, y / w, z / w)
         *
         * Converts from homogeneous 4D coordinates back to Cartesian 3D coordinates
         * by dividing all components by W.
         *
         * Why this exists:
         * In graphics, 3D positions are represented as 4D vectors with w=1.
         * After multiplication by a projection matrix, w is no longer 1.
         * The perspective divide converts these back to 3D screen coordinates.
         * This is what creates the effect of objects getting smaller with distance.
         *
         * The w component after projection contains the depth information.
         * Dividing by it maps the 3D frustum (truncated pyramid) to a rectangular box.
         *
         * Special case: Returns zero Vec3 if w = 0 to avoid division by zero.
         * w = 0 means the vector represents a direction, not a point.
         */
        [[nodiscard]] constexpr Vec3<T> PerspectiveDivide() const noexcept
        {
            if (w() == T(0))
            {
                return Vec3<T>();
            }

            return Vec3<T>(x() / w(), y() / w(), z() / w());
        }

        [[nodiscard]] constexpr Vec3<T> XYZ() const noexcept
        {
            return Vec3<T>(x(), y(), z());
        }

        [[nodiscard]] constexpr Vec2<T> XY() const noexcept
        {
            return Vec2<T>(x(), y());
        }

        /**
         * Clamp Color to Valid Range
         *
         * Clamps all components (R, G, B, A) to [0, 1] range.
         *
         * Why needed:
         * Color operations (addition, multiplication, lighting) can push
         * values outside the valid [0,1] range. GPUs and displays expect
         * colors in this range, so clamping is necessary before output.
         */
        [[nodiscard]] constexpr Vec4 ClampColor() const noexcept
        {
            return Vec4(
                std::clamp(r(), T(0), T(1)),
                std::clamp(g(), T(0), T(1)),
                std::clamp(b(), T(0), T(1)),
                std::clamp(a(), T(0), T(1))
            );
        }

        /**
         * Luminance (Perceived Brightness)
         *
         * Formula: L = 0.2126 * R + 0.7152 * G + 0.0722 * B
         *
         * Returns the perceived brightness of this color as a single scalar.
         * These coefficients come from the ITU-R BT.709 standard, which defines
         * how human eyes perceive different wavelengths of light.
         *
         * Why not just average RGB?
         * Human eyes are most sensitive to green light, less to red, and
         * least to blue. A simple average (R+G+B)/3 would not match
         * how we actually perceive brightness.
         *
         * Coefficients:
         *  - Green (0.7152): Eyes are most sensitive to green
         *  - Red   (0.2126): Moderate sensitivity
         *  - Blue  (0.0722): Least sensitive
         *  - Sum = 1.0 (white stays white)
         */
        [[nodiscard]] constexpr T Luminance() const noexcept
        {
            return T(0.2126) * r() + T(0.7152) * g() + T(0.0722) * b();
        }

        /**
         * Premultiplied Alpha
         *
         * Formula: result = (R * A, G * A, B * A, A)
         *
         * Converts this color to premultiplied alpha format by multiplying
         * each color channel by the alpha value.
         *
         * Why premultiplied alpha?
         * Standard alpha blending formula: result = src×srcA + dst×(1-srcA)
         * This requires two multiplications per pixel per blend.
         *
         * With premultiplied alpha: result = src + dst×(1-srcA)
         * Only one multiplication needed - faster for GPU blending.
         * Also handles transparent black correctly (avoids dark edges
         * around transparent sprites).
         */
        [[nodiscard]] constexpr Vec4 Premultiply() const noexcept
        {
            return Vec4(r() * a(), g() * a(), b() * a(), a());
        }

        /**
         * Unpremultiply Alpha
         *
         * Formula: result = (R / A, G / A, B / A, A)
         *
         * Reverses premultiplied alpha, recovering the original color values.
         *
         * Special case: Returns zero color if alpha = 0 (fully transparent),
         * since the original color is undefined when alpha is zero.
         */
        [[nodiscard]] constexpr Vec4 Unpremultiply() const noexcept
        {
            if (a() == T(0)) return Vec4();
            return Vec4(r() / a(), g() / a(), b() / a(), a());
        }

        /**
         * Swizzle - Reorder Components
         *
         * Creates a new Vec4 by selecting components in any order.
         *
         * In GLSL (shader language), you can do: vec4.zyxw
         * This provides the same functionality in C++.
         *
         * Parameters are indices into the data array:
         * - 0 = x/r, 1 = y/g, 2 = z/b, 3 = w/a
         */
        [[nodiscard]] constexpr Vec4 Swizzle(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const noexcept
        {
            if (i > 4 || j > 4 || k > 4 || l > 4)
            {
                return Vec4<T>();
            }

            return Vec4(Base::data[i], Base::data[j], Base::data[k], Base::data[l]);
        }
    };

    using Vec4f = Vec4<float>;
    using Vec4d = Vec4<double>;
    using Vec4i = Vec4<int>;
    using Vec4u = Vec4<unsigned int>;

    template<Arithmetic T>
    [[nodiscard]] constexpr T Dot(const Vec4<T>& a, const Vec4<T>& b) noexcept
    {
        return a.Dot(b);
    }

    template<Arithmetic T>
    [[nodiscard]] Vec4<T> Normalize(const Vec4<T>& v) noexcept
    {
        return v.Normalize();
    }

    template<Arithmetic T>
    [[nodiscard]] T Distance(const Vec4<T>& a, const Vec4<T>& b) noexcept
    {
        return a.Distance(b);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr Vec4<T> Lerp(const Vec4<T>& a, const Vec4<T>& b, T t) noexcept
    {
        return a.Lerp(b, t);
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr Vec3<T> PerspectiveDivide(const Vec4<T>& v) noexcept
    {
        return v.PerspectiveDivide();
    }

    template<Arithmetic T>
    [[nodiscard]] constexpr T Luminance(const Vec4<T>& color) noexcept
    {
        return color.Luminance();
    }

    template<Arithmetic T>
    inline constexpr Vec4<T> Vec4Zero = Vec4<T>(T(0), T(0), T(0), T(0));

    template<Arithmetic T>
    inline constexpr Vec4<T> Vec4One = Vec4<T>(T(1), T(1), T(1), T(1));

    template<Arithmetic T>
    inline constexpr Vec4<T> Vec4Right = Vec4<T>(T(1), T(0), T(0), T(0));

    template<Arithmetic T>
    inline constexpr Vec4<T> Vec4Up = Vec4<T>(T(0), T(1), T(0), T(0));

    template<Arithmetic T>
    inline constexpr Vec4<T> Vec4Forward = Vec4<T>(T(0), T(0), T(1), T(0));

    template<Arithmetic T>
    inline constexpr Vec4<T> ColorBlack = Vec4<T>(T(0), T(0), T(0), T(1));

    template<Arithmetic T>
    inline constexpr Vec4<T> ColorWhite = Vec4<T>(T(1), T(1), T(1), T(1));

    template<Arithmetic T>
    inline constexpr Vec4<T> ColorRed = Vec4<T>(T(1), T(0), T(0), T(1));

    template<Arithmetic T>
    inline constexpr Vec4<T> ColorGreen = Vec4<T>(T(0), T(1), T(0), T(1));

    template<Arithmetic T>
    inline constexpr Vec4<T> ColorBlue = Vec4<T>(T(0), T(0), T(1), T(1));

    template<Arithmetic T>
    inline constexpr Vec4<T> ColorTransparent = Vec4<T>(T(0), T(0), T(0), T(0));
} // namespace Pastel::Math