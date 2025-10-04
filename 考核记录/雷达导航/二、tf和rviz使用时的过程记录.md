1.不知道urdf是什么，通过gpt给出的示例，明白urdf是一种描述机器人结构的参数文件

2.完成urdf文件后，不知道怎么在rviz上显示，借助gpt学会通过file的方式可视化，gpt还给出了通过robot_state_publisher发布urdf文件，但是我在rviz中将source从file改为topic后并没有看见圆柱，然而我确认了`/robot_description`话题有正确urdf的内容，以及tf输出正确，其他的暂时没有思路。（又经过一番探索后找到了`Description Topic`的选项，选择`/robot_description`成功显示）



3.直接在命令行`ros2 run robot_state_publisher robot_state_publisher --ros-args   -p robot_description:=`无法加载太大的urdf

问gpt知道可以编写launch文件启动解决



4.无法显示完整的小车，只能看到一个圆柱体

gpt无法解决，遂上csdn找文档

`Global Options `→ `Fixed Frame` 设置为 `base_link`

5.然而还是少看到两个轮子，于是前往csdn找别人写的launch，发现大家还用了`joint_state_publisher_gui`，加上之后就能显示完整小车了

6.tf树不完整，少了两个轮子

检查`urdf`确认有从`base_link`到`wheek`的`joint`

询问gpt后发现是因为urdf文件中有的注释使用了`#`，而这是不规范的，导致有些东西没被正确加载，删除这些注释之后重新编译运行后得到完整的tf。这确实说明了在 URDF 里不能用 `#` 注释，它会被解析器当成错误字符，导致后续的 `<joint>` 没有被正确加载。