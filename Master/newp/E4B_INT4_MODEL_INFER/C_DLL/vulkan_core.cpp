#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <future>

// gemma's dictionary size = 256_000
#define MAX_M 262144
#define MAX_K 16384

struct PushConstants
{
    uint32_t M_out;
    uint32_t K_in_uints;
};

// 전역 Vulkan 핸들
VkInstance instance;
VkPhysicalDevice physicalDevice;
VkDevice device;
VkQueue computeQueue;
VkCommandPool commandPool;
VkPipeline computePipeline;
VkPipelineLayout pipelineLayout;
VkDescriptorSetLayout descriptorSetLayout;

//  핑퐁용 배열 2개
VkBuffer g_matBuf[2];
VkDeviceMemory g_matMem[2];
void *g_matMapped[2];
VkDescriptorSet g_descriptorSet[2];

// 단일 공유 버퍼
VkBuffer g_xBuf, g_scaleBuf, g_outBuf;
VkDeviceMemory g_xMem, g_scaleMem, g_outMem;
void *g_xMapped, *g_scaleMapped, *g_outMapped;
//  주의: 단일 g_descriptorSet 선언은 삭제함! (위에서 배열로 선언했으므로)

std::future<void> weight_loader;

std::vector<char> readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("cannot find shader file!");
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("cannot find memory type");
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory, void **mappedData)
{
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);
    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, memReq.size, findMemoryType(memReq.memoryTypeBits, properties)};
    vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
    vkMapMemory(device, bufferMemory, 0, size, 0, mappedData);
}

extern "C"
{
    void init_vulkan_engine()
    {
        std::cout << "[Vulkan] init engine.. " << std::endl;
        VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "Gemma3_NPU", 0, nullptr, 0, VK_API_VERSION_1_2};
        VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &appInfo, 0, nullptr, 0, nullptr};
        vkCreateInstance(&createInfo, nullptr, &instance);

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physicalDevice = devices[0];

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &queuePriority};
        VkDeviceCreateInfo deviceCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queueCreateInfo, 0, nullptr, 0, nullptr, nullptr};
        vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
        vkGetDeviceQueue(device, 0, 0, &computeQueue);

        VkDescriptorSetLayoutBinding bindings[4];
        for (int i = 0; i < 4; i++)
            bindings[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, bindings};
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);

        VkPushConstantRange pushConstantRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &descriptorSetLayout, 1, &pushConstantRange};
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

        auto shaderCode = readFile("C_DLL/gemv_int4_vector4.spv");
        VkShaderModuleCreateInfo shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, shaderCode.size(), reinterpret_cast<const uint32_t *>(shaderCode.data())};
        VkShaderModule computeShaderModule;
        vkCreateShaderModule(device, &shaderInfo, nullptr, &computeShaderModule);

        VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule, "main", nullptr}, pipelineLayout, VK_NULL_HANDLE, -1};
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);
        vkDestroyShaderModule(device, computeShaderModule, nullptr);

        VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkDeviceSize mat_buffer_size = 300 * 1024 * 1024; // 300MB

        for (int i = 0; i < 2; i++)
        {
            createBuffer(mat_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, g_matBuf[i], g_matMem[i], &g_matMapped[i]);
        }
        createBuffer(MAX_K * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, g_xBuf, g_xMem, &g_xMapped);
        createBuffer(MAX_M * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, g_scaleBuf, g_scaleMem, &g_scaleMapped);
        createBuffer(MAX_M * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, g_outBuf, g_outMem, &g_outMapped);

        VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
        VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 2, 1, &poolSize};
        VkDescriptorPool descriptorPool;
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);

        VkDescriptorSetLayout layouts[] = {descriptorSetLayout, descriptorSetLayout};
        VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 2, layouts};
        vkAllocateDescriptorSets(device, &allocInfo, g_descriptorSet);

        for (int i = 0; i < 2; i++)
        {
            VkDescriptorBufferInfo bufInfos[4] = {{g_xBuf, 0, VK_WHOLE_SIZE}, {g_matBuf[i], 0, VK_WHOLE_SIZE}, {g_scaleBuf, 0, VK_WHOLE_SIZE}, {g_outBuf, 0, VK_WHOLE_SIZE}};
            VkWriteDescriptorSet descriptorWrites[4] = {};
            for (int j = 0; j < 4; j++)
                descriptorWrites[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, g_descriptorSet[i], (uint32_t)j, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufInfos[j], nullptr};
            vkUpdateDescriptorSets(device, 4, descriptorWrites, 0, nullptr);
        }

        VkCommandPoolCreateInfo cmdPoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
        vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool);

        std::cout << "[Vulkan] buffer setting complete" << std::endl;
    }

    void prefetch_weight_async(const uint8_t *mat_p, int M_out, int K_in, int buf_idx)
    {
        weight_loader = std::async(std::launch::async, [=]()
                                   { memcpy(g_matMapped[buf_idx], mat_p, M_out * (K_in / 2) * sizeof(uint8_t)); });
    }

    void run_vulkan_gemv_pingpong(const float *x, const float *scale, float *out, int M_out, int K_in, int buf_idx)
    {
        if (weight_loader.valid())
        {
            weight_loader.wait();
        }

        memcpy(g_xMapped, x, K_in * sizeof(float));
        memcpy(g_scaleMapped, scale, M_out * sizeof(float));

        VkCommandBufferAllocateInfo cmdAllocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &g_descriptorSet[buf_idx], 0, nullptr);

        PushConstants pushParams = {(uint32_t)M_out, (uint32_t)(K_in / 32)};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushParams);

        uint32_t groupCountX = (M_out + 31) / 32;
        vkCmdDispatch(commandBuffer, groupCountX, 1, 1);

        //  주의: 여기서 memcpy(out, g_outMapped...) 하는 오류를 삭제함!

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &commandBuffer, 0, nullptr};
        vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(computeQueue);

        //  계산 완료 후 여기서 딱 한 번만 결과를 뺌!
        memcpy(out, g_outMapped, M_out * sizeof(float));

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    // 호환성을 위해 남겨둔 레거시 함수 (배열 인덱스 [0]으로 수정)
    void run_vulkan_gemv(const float *x, const uint8_t *mat_p, const float *scale, float *out, int M_out, int K_in)
    {
        memcpy(g_xMapped, x, K_in * sizeof(float));
        memcpy(g_matMapped[0], mat_p, M_out * (K_in / 2) * sizeof(uint8_t)); //  [0] 사용
        memcpy(g_scaleMapped, scale, M_out * sizeof(float));

        VkCommandBufferAllocateInfo cmdAllocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &g_descriptorSet[0], 0, nullptr); //  [0] 사용

        PushConstants pushParams;
        pushParams.M_out = (uint32_t)M_out;
        pushParams.K_in_uints = (uint32_t)(K_in / 32);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushParams);

        uint32_t groupCountX = (M_out + 31) / 32;
        vkCmdDispatch(commandBuffer, groupCountX, 1, 1);
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &commandBuffer, 0, nullptr};
        vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(computeQueue);

        memcpy(out, g_outMapped, M_out * sizeof(float));
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
}