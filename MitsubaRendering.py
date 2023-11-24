# import numpy as np
# import mitsuba as mi

# mi.set_variant("cuda_ad_rgb")

# xml_head = \
#     """
# <scene version="0.6.0">
#     <integrator type="path">
#         <integer name="maxDepth" value="-1"/>
#     </integrator>
#     <sensor type="perspective">
#         <float name="farClip" value="100"/>
#         <float name="nearClip" value="0.1"/>
#         <transform name="toWorld">
#             <lookat origin="-3,3,3" target="0,0,0" up="0,0,1"/>
#         </transform>
#         <float name="fov" value="18"/>
#         <sampler type="independent">
#             <integer name="sampleCount" value="256"/>
#         </sampler>
#         <film type="hdrfilm">
#             <integer name="width" value="1080"/>
#             <integer name="height" value="1080"/>
#             <rfilter type="gaussian"/>
#         </film>
#     </sensor>
    
#     <bsdf type="roughplastic" id="surfaceMaterial">
#         <string name="distribution" value="ggx"/>
#         <float name="alpha" value="0.05"/>
#         <float name="intIOR" value="1.46"/>
#         <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
#     </bsdf>
    
# """

# # I also use a smaller point size
# xml_ball_segment = \
#     """
#     <shape type="sphere">
#         <float name="radius" value="0.015"/>
#         <transform name="toWorld">
#             <translate x="{}" y="{}" z="{}"/>
#         </transform>
#         <bsdf type="diffuse">
#             <rgb name="reflectance" value="{},{},{}"/>
#         </bsdf>
#     </shape>
# """

# xml_tail = \
#     """
#     <shape type="rectangle">
#         <ref name="bsdf" id="surfaceMaterial"/>
#         <transform name="toWorld">
#             <scale x="10" y="10" z="1"/>
#             <translate x="0" y="0" z="-0.5"/>
#         </transform>
#     </shape>
    
#     <shape type="rectangle">
#         <transform name="toWorld">
#             <scale x="10" y="10" z="1"/>
#             <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
#         </transform>
#         <emitter type="area">
#             <rgb name="radiance" value="6,6,6"/>
#         </emitter>
#     </shape>
# </scene>
# """


# def colormap(x, y, z):
#     vec = np.array([x, y, z])
#     vec = np.clip(vec, 0.001, 1.0)
#     norm = np.sqrt(np.sum(vec ** 2))
#     vec /= norm
#     return [vec[0], vec[1], vec[2]]


# def standardize_bbox(pcl, points_per_object):
#     pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
#     np.random.shuffle(pt_indices)
#     pcl = pcl[pt_indices]  # n by 3
#     mins = np.amin(pcl, axis=0)
#     maxs = np.amax(pcl, axis=0)
#     center = (mins + maxs) / 2.
#     scale = np.amax(maxs - mins)
#     # print("Center: {}, Scale: {}".format(center, scale))
#     result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
#     return result


# # only for debugging reasons
# def writeply(vertices, ply_file):
#     sv = np.shape(vertices)
#     points = []
#     for v in range(sv[0]):
#         vertex = vertices[v]
#         points.append("%f %f %f\n" % (vertex[0], vertex[1], vertex[2]))
#     print(np.shape(points))
#     file = open(ply_file, "w")
#     file.write('''ply
#     format ascii 1.0
#     element vertex %d
#     property float x
#     property float y
#     property float z
#     end_header
#     %s
#     ''' % (len(vertices), "".join(points)))
#     file.close()

# def ImageFromNumpyArr(np_arr):
#     pclTime = np_arr
#     pclTimeSize = np.shape(pclTime)

#     if (len(np.shape(pclTime)) < 3):
#         pclTimeSize = [1, np.shape(pclTime)[0], np.shape(pclTime)[1]]
#         pclTime.resize(pclTimeSize)

#     for pcli in range(0, pclTimeSize[0]):
#         pcl = pclTime[pcli, :, :]

#         pcl = standardize_bbox(pcl, 1024)
#         pcl = pcl[:, [2, 0, 1]]
#         pcl[:, 0] *= -1
#         pcl[:, 2] += 0.0125

#         xml_segments = [xml_head]
#         for i in range(pcl.shape[0]):
#             color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
#             xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
#         xml_segments.append(xml_tail)

#         xml_content = str.join('', xml_segments)

#         scene = mi.load_string(xml_content)

#         image = mi.render(scene=scene)
#         img = np.array(image)
#         img = (img)**(1 / 2)

#         return img