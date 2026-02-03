#include <numbers>

#include "PastelMath/Vector/Vec2.hpp"
#include "PastelMath/Vector/Vec3.hpp"
#include "PastelMath/Vector/Vec4.hpp"

#include "gtest/gtest.h"

using namespace Pastel::Math;

constexpr double PI = std::numbers::pi;

TEST(Vec2Test, Construction)
{
    Vec2f v1;
    EXPECT_FLOAT_EQ(v1.x(), 0.0f);
    EXPECT_FLOAT_EQ(v1.y(), 0.0f);

    Vec2f v2(3.0f, 4.0f);
    EXPECT_FLOAT_EQ(v2.x(), 3.0f);
    EXPECT_FLOAT_EQ(v2.y(), 4.0f);
}

TEST(Vec2Test, NamedAccessors)
{
    Vec2f v(1.0f, 2.0f);

    EXPECT_FLOAT_EQ(v.x(), 1.0f);
    EXPECT_FLOAT_EQ(v.y(), 2.0f);

    EXPECT_FLOAT_EQ(v.u(), 1.0f);
    EXPECT_FLOAT_EQ(v.v(), 2.0f);

    EXPECT_FLOAT_EQ(v.s(), 1.0f);
    EXPECT_FLOAT_EQ(v.t(), 2.0f);

    v.x() = 5.0f;
    EXPECT_FLOAT_EQ(v.u(), 5.0f);
    EXPECT_FLOAT_EQ(v.s(), 5.0f);
}

TEST(Vec2Test, Magnitude)
{
    Vec2f v(3.0f, 4.0f);

    EXPECT_FLOAT_EQ(v.MagnitudeSq(), 25.0f);
    EXPECT_FLOAT_EQ(v.Magnitude(), 5.0f);
}

TEST(Vec2Test, Normalize)
{
    Vec2f v(3.0f, 4.0f);
    Vec2f n = v.Normalize();

    EXPECT_FLOAT_EQ(n.x(), 0.6f);
    EXPECT_FLOAT_EQ(n.y(), 0.8f);
    EXPECT_NEAR(n.Magnitude(), 1.0f, 1e-6f);
}

TEST(Vec2Test, DotProduct)
{
    Vec2f a(1.0f, 2.0f);
    Vec2f b(3.0f, 4.0f);

    EXPECT_FLOAT_EQ(a.Dot(b), 11.0f);
    EXPECT_FLOAT_EQ(Dot(a, b), 11.0f);
}

TEST(Vec2Test, CrossProduct)
{
    Vec2f a(1.0f, 0.0f);
    Vec2f b(0.0f, 1.0f);

    EXPECT_FLOAT_EQ(a.Cross(b), 1.0f);
    EXPECT_FLOAT_EQ(b.Cross(a), -1.0f);
}

TEST(Vec2Test, Perpendicular)
{
    Vec2f v(3.0f, 4.0f);
    Vec2f perp = v.Perp();

    EXPECT_FLOAT_EQ(perp.x(), -4.0f);
    EXPECT_FLOAT_EQ(perp.y(), 3.0f);

    EXPECT_NEAR(v.Dot(perp), 0.0f, 1e-6f);
}

TEST(Vec2Test, Angle)
{
    Vec2f right(1.0f, 0.0f);
    Vec2f up(0.0f, 1.0f);

    EXPECT_NEAR(right.Angle(), 0.0f, 1e-6f);
    EXPECT_NEAR(up.Angle(), PI / 2.0, 1e-6f);
}

TEST(Vec2Test, Rotation)
{
    Vec2f v(1.0f, 0.0f);
    Vec2f rotated = v.Rotate(static_cast<float>(PI / 2.0));

    EXPECT_NEAR(rotated.x(), 0.0f, 1e-6f);
    EXPECT_NEAR(rotated.y(), 1.0f, 1e-6f);
}

TEST(Vec2Test, Arithmetic)
{
    Vec2f a(1.0f, 2.0f);
    Vec2f b(3.0f, 4.0f);

    Vec2f sum = a + b;
    EXPECT_FLOAT_EQ(sum.x(), 4.0f);
    EXPECT_FLOAT_EQ(sum.y(), 6.0f);

    Vec2f diff = b - a;
    EXPECT_FLOAT_EQ(diff.x(), 2.0f);
    EXPECT_FLOAT_EQ(diff.y(), 2.0f);

    Vec2f scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled.x(), 2.0f);
    EXPECT_FLOAT_EQ(scaled.y(), 4.0f);
}

TEST(Vec2Test, Distance)
{
    Vec2f a(0.0f, 0.0f);
    Vec2f b(3.0f, 4.0f);

    EXPECT_FLOAT_EQ(a.Distance(b), 5.0f);
    EXPECT_FLOAT_EQ(Distance(a, b), 5.0f);
}

TEST(Vec2Test, Lerp)
{
    Vec2f a(0.0f, 0.0f);
    Vec2f b(10.0f, 10.0f);

    Vec2f mid = a.Lerp(b, 0.5f);
    EXPECT_FLOAT_EQ(mid.x(), 5.0f);
    EXPECT_FLOAT_EQ(mid.y(), 5.0f);

    Vec2f quarter = Lerp(a, b, 0.25f);
    EXPECT_FLOAT_EQ(quarter.x(), 2.5f);
    EXPECT_FLOAT_EQ(quarter.y(), 2.5f);
}

TEST(Vec2Test, ApproxEqual)
{
    Vec2f a(1.0f, 2.0f);
    Vec2f b(1.0000001f, 2.0000001f);

    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a.ApproxEqual(b));
}

TEST(Vec3Test, Construction)
{
    Vec3f v1;
    EXPECT_FLOAT_EQ(v1.x(), 0.0f);
    EXPECT_FLOAT_EQ(v1.y(), 0.0f);
    EXPECT_FLOAT_EQ(v1.z(), 0.0f);

    Vec3f v2(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v2.x(), 1.0f);
    EXPECT_FLOAT_EQ(v2.y(), 2.0f);
    EXPECT_FLOAT_EQ(v2.z(), 3.0f);
}

TEST(Vec3Test, NamedAccessors)
{
    Vec3f v(1.0f, 2.0f, 3.0f);

    EXPECT_FLOAT_EQ(v.x(), 1.0f);
    EXPECT_FLOAT_EQ(v.y(), 2.0f);
    EXPECT_FLOAT_EQ(v.z(), 3.0f);

    EXPECT_FLOAT_EQ(v.r(), 1.0f);
    EXPECT_FLOAT_EQ(v.g(), 2.0f);
    EXPECT_FLOAT_EQ(v.b(), 3.0f);
}

TEST(Vec3Test, CrossProduct)
{
    Vec3f x(1.0f, 0.0f, 0.0f);
    Vec3f y(0.0f, 1.0f, 0.0f);
    Vec3f z = x.Cross(y);

    EXPECT_FLOAT_EQ(z.x(), 0.0f);
    EXPECT_FLOAT_EQ(z.y(), 0.0f);
    EXPECT_FLOAT_EQ(z.z(), 1.0f);

    Vec3f zNeg = y.Cross(x);
    EXPECT_FLOAT_EQ(zNeg.z(), -1.0f);
}

TEST(Vec3Test, ScalarTriple)
{
    Vec3f a(1.0f, 0.0f, 0.0f);
    Vec3f b(0.0f, 1.0f, 0.0f);
    Vec3f c(0.0f, 0.0f, 1.0f);

    EXPECT_FLOAT_EQ(a.ScalarTriple(b, c), 1.0f);
}

TEST(Vec3Test, Magnitude)
{
    Vec3f v(1.0f, 2.0f, 2.0f);
    EXPECT_FLOAT_EQ(v.MagnitudeSq(), 9.0f);
    EXPECT_FLOAT_EQ(v.Magnitude(), 3.0f);
}

TEST(Vec3Test, SphericalCoordinates)
{
    Vec3f v = Vec3f::FromSpherical(0.0f, 0.0f, 1.0f);
    EXPECT_NEAR(v.x(), 1.0f, 1e-6f);
    EXPECT_NEAR(v.y(), 0.0f, 1e-6f);
    EXPECT_NEAR(v.z(), 0.0f, 1e-6f);

    Vec3f up = Vec3f::FromSpherical(0.0f, static_cast<float>(PI / 2.0), 1.0f);
    EXPECT_NEAR(up.z(), 1.0f, 1e-6f);
}

TEST(Vec3Test, Constants)
{
    EXPECT_FLOAT_EQ(Vec3fRight.x(), 1.0f);
    EXPECT_FLOAT_EQ(Vec3fUp.y(), 1.0f);
    EXPECT_FLOAT_EQ(Vec3fForward.z(), 1.0f);
}

TEST(Vec4Test, Construction)
{
    Vec4f v1;
    EXPECT_FLOAT_EQ(v1.x(), 0.0f);
    EXPECT_FLOAT_EQ(v1.w(), 0.0f);

    Vec4f v2(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_FLOAT_EQ(v2.x(), 1.0f);
    EXPECT_FLOAT_EQ(v2.y(), 2.0f);
    EXPECT_FLOAT_EQ(v2.z(), 3.0f);
    EXPECT_FLOAT_EQ(v2.w(), 4.0f);
}

TEST(Vec4Test, Vec3Constructor)
{
    Vec3f v3(1.0f, 2.0f, 3.0f);
    Vec4f v4(v3, 5.0f);

    EXPECT_FLOAT_EQ(v4.x(), 1.0f);
    EXPECT_FLOAT_EQ(v4.y(), 2.0f);
    EXPECT_FLOAT_EQ(v4.z(), 3.0f);
    EXPECT_FLOAT_EQ(v4.w(), 5.0f);
}

TEST(Vec4Test, PerspectiveDivide)
{
    Vec4f v(8.0f, 4.0f, 2.0f, 2.0f);
    Vec3f result = v.PerspectiveDivide();

    EXPECT_FLOAT_EQ(result.x(), 4.0f);
    EXPECT_FLOAT_EQ(result.y(), 2.0f);
    EXPECT_FLOAT_EQ(result.z(), 1.0f);
}

TEST(Vec4Test, XYZExtraction)
{
    Vec4f v(1.0f, 2.0f, 3.0f, 4.0f);
    Vec3f xyz = v.XYZ();

    EXPECT_FLOAT_EQ(xyz.x(), 1.0f);
    EXPECT_FLOAT_EQ(xyz.y(), 2.0f);
    EXPECT_FLOAT_EQ(xyz.z(), 3.0f);
}

TEST(Vec4Test, ColorOperations)
{
    Vec4f color(0.5f, 0.7f, 0.3f, 0.5f);

    Vec4f premult = color.Premultiply();
    EXPECT_FLOAT_EQ(premult.r(), 0.25f);
    EXPECT_FLOAT_EQ(premult.g(), 0.35f);
    EXPECT_FLOAT_EQ(premult.b(), 0.15f);
    EXPECT_FLOAT_EQ(premult.a(), 0.5f);

    Vec4f unpremult = premult.Unpremultiply();
    EXPECT_NEAR(unpremult.r(), 0.5f, 1e-6f);
    EXPECT_NEAR(unpremult.g(), 0.7f, 1e-6f);
    EXPECT_NEAR(unpremult.b(), 0.3f, 1e-6f);
}

TEST(Vec4Test, Luminance)
{
    Vec4f white(1.0f, 1.0f, 1.0f, 1.0f);
    EXPECT_NEAR(white.Luminance(), 1.0f, 1e-6f);

    Vec4f red(1.0f, 0.0f, 0.0f, 1.0f);
    EXPECT_NEAR(red.Luminance(), 0.2126f, 1e-4f);

    Vec4f green(0.0f, 1.0f, 0.0f, 1.0f);
    EXPECT_NEAR(green.Luminance(), 0.7152f, 1e-4f);
}

TEST(Vec4Test, ColorConstants)
{
    EXPECT_FLOAT_EQ(ColorWhite<float>.r(), 1.0f);
    EXPECT_FLOAT_EQ(ColorWhite<float>.a(), 1.0f);

    EXPECT_FLOAT_EQ(ColorBlack<float>.r(), 0.0f);
    EXPECT_FLOAT_EQ(ColorBlack<float>.a(), 1.0f);

    EXPECT_FLOAT_EQ(ColorRed<float>.r(), 1.0f);
    EXPECT_FLOAT_EQ(ColorRed<float>.g(), 0.0f);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}