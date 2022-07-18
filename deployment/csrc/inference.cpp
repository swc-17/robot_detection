#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
using namespace std;

#include <cuda.h> 
#include <cuda_runtime.h> 


#include <NvInfer.h>
#include <NvInferPlugin.h>
using namespace nvinfer1;

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>


class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


// IRuntime* runtime = createInferRuntime(logger);
// ICudaEngine* engine = 
//     runtime->deserializeCudaEngine(modelData, modelSize);
// IExecutionContext *context = engine->createExecutionContext();
// int32_t inputIndex = engine->getBindingIndex(INPUT_NAME);
// int32_t outputIndex = engine->getBindingIndex(OUTPUT_NAME);
// void* buffers[2];
// buffers[inputIndex] = inputBuffer;
// buffers[outputIndex] = outputBuffer;
// context->enqueueV2(buffers, stream, nullptr);

int main(int argc, char** argv)
{
    // init plugins
    initLibNvInferPlugins(&logger,"");
    // read plan file
    std::string cached_path = "/home/sunwenchao/robot/models/tiny.trt";
    std::ifstream fin(cached_path);
    std::string cached_engine = "";
    while (fin.peek() != EOF){ 
            std::stringstream buffer;
            buffer << fin.rdbuf();
            cached_engine.append(buffer.str());
    }
    fin.close();
    // build runtime 
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = 
        runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    // set input and output buffer
    int n = engine->getNbBindings();
    vector<int> index;
    Dims dim;
    for(int i=0;i<n;i++){
        index.push_back(i);
        dim = engine->getBindingDimensions(i);
        cout << dim.d[0] << endl;
        cout << dim.d[1] << endl;
        cout << dim.d[2] << endl;
        cout << dim.d[3] << endl;
        cout << endl;
    } 

    float* input[3 *416 * 416];
    float* output[2];

    void* buffers[n];
    cudaMalloc(&buffers[0], 1* 3 *416 * 416 * sizeof(float));
    cudaMalloc(&buffers[1], 1* 2 * sizeof(float));
    cudaStream_t stream;
	cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[0], input, 
        1* 3 *416 * 416 * sizeof(float), 
        cudaMemcpyHostToDevice, stream);
    // context->enqueueV2(buffers, stream, nullptr); 
    context->executeV2(buffers);
    cudaMemcpyAsync(output, buffers[1], 
        2 * sizeof(float), 
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}



// void doInference(IExecutionContext& context, float* input, float** output, int batchSize)
// {
// 	const ICudaEngine& engine = context.getEngine();
// 	assert(engine.getNbBindings() == 3);
// 	void* buffers[3];
// 	const int inputIndex = 0;
// 	const int outputLocIndex = 1;
// 	const int outputConfIndex = 2;
// 	// create GPU buffers, 申请GPU显存, Allocate GPU memory for Input / Output data
// 	cudaMalloc(&buffers[inputIndex], batchSize * INPUT_CHANNEL * DETECT_HEIGHT * DETECT_WIDTH * sizeof(float));
//     cudaMalloc(&buffers[outputLocIndex], batchSize * 12168 * sizeof(float));
// 	cudaMalloc(&buffers[outputConfIndex], batchSize * 3042 * sizeof(float));
// 	//使用cuda 流来管理并行计算, Use CUDA streams to manage the concurrency of copying and executing
// 	cudaStream_t stream;
// 	CHECK(cudaStreamCreate(&stream));
// 	//内存到显存，input是读入内存中的数据；buffers[inputIndex]是显存上的存储区域，用于存放输入数据
// 	// Copy Input Data to the GPU
//     cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_CHANNEL * DETECT_HEIGHT * DETECT_WIDTH * sizeof(float), cudaMemcpyHostToDevice, stream);
// 	context.enqueue(batchSize, buffers, stream, nullptr);
// 	cudaMemcpyAsync(output[0], buffers[outputLocIndex], batchSize * 12168 * sizeof(float), cudaMemcpyDeviceToHost, stream);
// 	cudaMemcpyAsync(output[1], buffers[outputConfIndex], batchSize * 3042 * sizeof(float), cudaMemcpyDeviceToHost, stream);
// 	//如果使用了多个cuda流，需要同步
// 	cudaStreamSynchronize(stream);
// 	// release the stream and the buffers
// 	cudaStreamDestroy(stream);
// 	cudaFree(buffers[inputIndex]);
// 	cudaFree(buffers[outputLocIndex]);
// 	cudaFree(buffers[outputConfIndex]);
// }

// # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
// #       with open("/home/swc/CenterNet/src/model-fp32.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
// # 		      engine = runtime.deserialize_cuda_engine(f.read())
// #       h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
// #       h_output = [cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(i)), dtype=np.float32) for i in range(1,7)]
// #       # Allocate device memory for inputs and outputs.
// #       d_input = cuda.mem_alloc(h_input.nbytes)
// #       d_output = [cuda.mem_alloc(h_output[i].nbytes) for i in range(6)]
// #       # Create a stream in which to copy inputs/outputs and run inference.
// #       stream = cuda.Stream()
// #       bindings = [int(d_output[i]) for i in range(6)]
// #       bindings.insert(0,int(d_input))

// #       with engine.create_execution_context() as context:
// #           # Transfer input data to the GPU.
// #           cuda.memcpy_htod_async(d_input, h_input, stream)
// #           # Run inference.
// #           context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
// #           # Transfer predictions back from the GPU.
// #           [cuda.memcpy_dtoh_async(h_output[i], d_output[i], stream) for i in range(6)]
// #           # Synchronize the stream
// #           stream.synchronize()