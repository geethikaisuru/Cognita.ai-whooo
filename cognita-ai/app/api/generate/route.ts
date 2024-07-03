import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import { Readable } from 'stream'
import { join } from 'path'
import fs from 'fs/promises'
import os from 'os'

export async function POST(req: NextRequest) {
  const formData = await req.formData()
  const files = formData.getAll('file') as File[]

  const tempDir = await fs.mkdtemp(join(os.tmpdir(), 'paper-generator-'))

  try {
    const filePaths = await Promise.all(
      files.map(async (file, index) => {
        const filePath = join(tempDir, `input${index}.pdf`)
        await fs.writeFile(filePath, Buffer.from(await file.arrayBuffer()))
        return filePath
      })
    )

    const pythonProcess = spawn('python', ['PGen2.py', ...filePaths])

    const stream = new ReadableStream({
      start(controller) {
        pythonProcess.stdout.on('data', (data) => {
          controller.enqueue(data)
        })
        pythonProcess.stderr.on('data', (data) => {
          controller.enqueue(data)
        })
        pythonProcess.on('close', () => {
          controller.close()
        })
      },
    })

    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked',
      },
    })
  } catch (error) {
    console.error('Error:', error)
    return NextResponse.json({ error: 'An error occurred during generation.' }, { status: 500 })
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true })
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
}