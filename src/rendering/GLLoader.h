#pragma once
#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <GL/gl.h>
#else
#include <GL/gl.h>
#endif

// Constants for Modern GL (3.3+)
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_ARRAY_BUFFER 0x8892
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_FLOAT 0x1406
#define GL_FALSE 0
#define GL_TRUE 1
#define GL_DEPTH_TEST 0x0B71

typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;

// Function pointers typedefs
extern "C" {
    typedef void (*PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *buffers);
    typedef void (*PFNGLBINDBUFFERPROC) (GLenum target, GLuint buffer);
    typedef void (*PFNGLBUFFERDATAPROC) (GLenum target, GLsizeiptr size, const void *data, GLenum usage);
    typedef void (*PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *arrays);
    typedef void (*PFNGLBINDVERTEXARRAYPROC) (GLuint array);
    typedef void (*PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint index);
    typedef void (*PFNGLVERTEXATTRIBPOINTERPROC) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
    typedef void (*PFNGLDRAWARRAYSPROC) (GLenum mode, GLint first, GLsizei count);
    typedef GLuint (*PFNGLCREATESHADERPROC) (GLenum type);
    typedef void (*PFNGLSHADERSOURCEPROC) (GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
    typedef void (*PFNGLCOMPILESHADERPROC) (GLuint shader);
    typedef void (*PFNGLGETSHADERIVPROC) (GLuint shader, GLenum pname, GLint *params);
    typedef void (*PFNGLGETSHADERINFOLOGPROC) (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
    typedef GLuint (*PFNGLCREATEPROGRAMPROC) (void);
    typedef void (*PFNGLATTACHSHADERPROC) (GLuint program, GLuint shader);
    typedef void (*PFNGLLINKPROGRAMPROC) (GLuint program);
    typedef void (*PFNGLGETPROGRAMIVPROC) (GLuint program, GLenum pname, GLint *params);
    typedef void (*PFNGLGETPROGRAMINFOLOGPROC) (GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
    typedef void (*PFNGLUSEPROGRAMPROC) (GLuint program);
    typedef GLint (*PFNGLGETUNIFORMLOCATIONPROC) (GLuint program, const GLchar *name);
    typedef void (*PFNGLUNIFORMMATRIX4FVPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
}

namespace WulfNet {
namespace Rendering {

bool initGL();

extern PFNGLGENBUFFERSPROC glGenBuffers;
extern PFNGLBINDBUFFERPROC glBindBuffer;
extern PFNGLBUFFERDATAPROC glBufferData;
extern PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
extern PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
extern PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
extern PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
extern PFNGLDRAWARRAYSPROC glDrawArrays;
extern PFNGLCREATESHADERPROC glCreateShader;
extern PFNGLSHADERSOURCEPROC glShaderSource;
extern PFNGLCOMPILESHADERPROC glCompileShader;
extern PFNGLGETSHADERIVPROC glGetShaderiv;
extern PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
extern PFNGLCREATEPROGRAMPROC glCreateProgram;
extern PFNGLATTACHSHADERPROC glAttachShader;
extern PFNGLLINKPROGRAMPROC glLinkProgram;
extern PFNGLGETPROGRAMIVPROC glGetProgramiv;
extern PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
extern PFNGLUSEPROGRAMPROC glUseProgram;
extern PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
extern PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv;

} // namespace Rendering
} // namespace WulfNet
